from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from math import prod
from typing import TypeAlias

import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)

KVCacheValue: TypeAlias = torch.Tensor | Sequence[torch.Tensor]


class NPUOffloadingConnectorWorker(OffloadingConnectorWorker):
    """Worker-side offloading adapter for Ascend KV-cache layouts.

    vLLM's upstream offloading worker expects one tensor per attention layer
    before it canonicalizes KV caches. Ascend backends commonly expose a layer
    as separate K/V tensors, and some sparse/MLA paths expose extra per-layer
    tensors. Canonicalize those tensors here and then reuse the upstream
    offloading worker flow.
    """

    def _as_block_tensor(
        self,
        tensor: torch.Tensor,
        page_size_bytes: int,
    ) -> torch.Tensor:
        elem_size = tensor.element_size()
        byte_offset = tensor.storage_offset() * elem_size
        block_stride_bytes = tensor.stride(0) * elem_size
        expected_stride = 1
        for size, stride in zip(reversed(tensor.shape[1:]), reversed(tensor.stride()[1:])):
            if size > 1 and stride != expected_stride:
                raise ValueError(
                    "Cannot offload an Ascend KV cache whose block payload "
                    f"is non-contiguous: shape={tuple(tensor.shape)}, "
                    f"stride={tensor.stride()}"
                )
            expected_stride *= size
        if block_stride_bytes < page_size_bytes:
            raise ValueError(
                "Ascend KV cache blocks overlap in storage: "
                f"stride_bytes={block_stride_bytes}, "
                f"page_size_bytes={page_size_bytes}"
            )
        return torch.tensor(
            [],
            dtype=torch.int8,
            device=tensor.device,
        ).set_(
            tensor.untyped_storage(),
            byte_offset,
            (self.spec.kv_cache_config.num_blocks, page_size_bytes),
            (block_stride_bytes, 1),
        )

    @staticmethod
    def _tensor_page_size_bytes(tensor: torch.Tensor) -> int:
        return prod(tensor.shape[1:]) * tensor.element_size()

    def _add_tensor_ref(
        self,
        tensor: torch.Tensor,
        block_tensors: list[CanonicalKVCacheTensor],
        tensor_to_idx: dict[tuple[int, int, int, torch.device], int],
        ref_page_size_bytes: int | None = None,
    ) -> CanonicalKVCacheRef:
        tensor_page_size_bytes = self._tensor_page_size_bytes(tensor)
        block_tensor = self._as_block_tensor(tensor, tensor_page_size_bytes)
        key = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset() * tensor.element_size(),
            block_tensor.stride(0),
            tensor.device,
        )
        tensor_idx = tensor_to_idx.get(key)
        if tensor_idx is None:
            tensor_idx = len(block_tensors)
            tensor_to_idx[key] = tensor_idx
            block_tensors.append(
                CanonicalKVCacheTensor(
                    tensor=block_tensor,
                    page_size_bytes=tensor_page_size_bytes,
                )
            )

        return CanonicalKVCacheRef(
            tensor_idx=tensor_idx,
            page_size_bytes=(
                ref_page_size_bytes if ref_page_size_bytes is not None else self._tensor_page_size_bytes(tensor)
            ),
        )

    def register_kv_caches(self, kv_caches: dict[str, KVCacheValue]):
        # Preserve upstream v0.25 canonicalization (including packed-layout
        # handling and stride fixes) whenever the cache layout is already
        # upstream-compatible. Ascend-specific handling is only needed for
        # split attention tensors and separately allocated Mamba states.
        requires_layout_adaptation = False
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_spec = kv_cache_group.kv_cache_spec
            per_layer_specs = group_spec.kv_cache_specs if isinstance(group_spec, UniformTypeKVCacheSpecs) else {}
            for layer_name in kv_cache_group.layer_names:
                layer_spec = per_layer_specs.get(layer_name, group_spec)
                layer_cache = kv_caches[layer_name]
                if (isinstance(layer_spec, AttentionSpec) and not isinstance(layer_cache, torch.Tensor)) or isinstance(
                    layer_spec, MambaSpec
                ):
                    requires_layout_adaptation = True
                    break
            if requires_layout_adaptation:
                break

        if not requires_layout_adaptation:
            super().register_kv_caches(kv_caches)  # type: ignore[arg-type]
            return

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        tensor_to_idx: dict[tuple[int, int, int, torch.device], int] = {}

        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            per_layer_specs = (
                group_kv_cache_spec.kv_cache_specs if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs) else {}
            )

            for layer_name in kv_cache_group.layer_names:
                layer_kv_cache_spec = per_layer_specs.get(layer_name, group_kv_cache_spec)
                layer_kv_cache = kv_caches[layer_name]

                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    if isinstance(layer_kv_cache, torch.Tensor):
                        block_data_refs[layer_name].append(
                            self._add_tensor_ref(
                                layer_kv_cache,
                                block_tensors,
                                tensor_to_idx,
                                ref_page_size_bytes=(layer_kv_cache_spec.real_page_size_bytes),
                            )
                        )
                    else:
                        component_bytes = sum(self._tensor_page_size_bytes(tensor) for tensor in layer_kv_cache)
                        if component_bytes != layer_kv_cache_spec.real_page_size_bytes:
                            raise ValueError(
                                "Ascend attention KV cache components do not "
                                f"cover one logical page for {layer_name}: "
                                f"components={component_bytes}, "
                                f"spec={layer_kv_cache_spec.real_page_size_bytes}"
                            )
                        for tensor in layer_kv_cache:
                            block_data_refs[layer_name].append(
                                self._add_tensor_ref(
                                    tensor,
                                    block_tensors,
                                    tensor_to_idx,
                                )
                            )

                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    assert isinstance(layer_kv_cache, list)
                    assert len(layer_kv_cache) > 0
                    state_bytes = sum(self._tensor_page_size_bytes(tensor) for tensor in layer_kv_cache)
                    expected_bytes = replace(layer_kv_cache_spec, page_size_padded=None).page_size_bytes
                    if state_bytes != expected_bytes:
                        raise ValueError(
                            f"Mamba KV cache page size mismatch for {layer_name}: "
                            f"tensors={state_bytes}, spec={expected_bytes}"
                        )
                    # Ascend state tensors may be separately allocated or have
                    # aligned, non-zero storage offsets. Register each state
                    # independently instead of reconstructing a single tensor
                    # from the first state's entire storage.
                    for tensor in layer_kv_cache:
                        block_data_refs[layer_name].append(
                            self._add_tensor_ref(
                                tensor,
                                block_tensors,
                                tensor_to_idx,
                            )
                        )
                else:
                    raise NotImplementedError

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        self._init_worker(
            CanonicalKVCaches(
                tensors=block_tensors,
                group_data_refs=group_data_refs,
            )
        )


class NPUOffloadingConnector(OffloadingConnector):
    # ``OffloadingConnector`` already sets this, but it comes from an unfollowed
    # vLLM import (mypy runs with ``--follow-imports skip``), so its inherited
    # type is undeterminable; redeclare it here so mypy can resolve reads and
    # writes of ``self.connector_worker`` ("Cannot determine type of ...").
    connector_worker: OffloadingConnectorWorker | None

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            base_worker = self.connector_worker
            assert base_worker is not None
            self.connector_worker = NPUOffloadingConnectorWorker(base_worker.spec)

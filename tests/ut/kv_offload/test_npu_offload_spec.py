# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression tests for the Ascend NPU KV offloading specs.

These guard against the worker-side specs drifting from the upstream
``CPUOffloadingSpec`` / ``TieringOffloadingSpec`` / ``SharedOffloadRegion``
APIs. Upstream reworked ``SharedOffloadRegion.__init__`` to size the mmap from
``num_blocks * kv_bytes_per_block`` and dropped the old ``total_size_bytes`` /
``num_workers`` kwargs; the multi-tier offloading adaptation must follow that
signature or it crashes at worker-side handler construction time.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_offload.base import OffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

import vllm_ascend.kv_offload.npu as npu_mod
from vllm_ascend.distributed.kv_transfer.offloading_connector import (
    NPUOffloadingConnectorWorker,
)


def test_npu_worker_implements_v025_worker_protocol():
    assert issubclass(npu_mod.NPUOffloadingWorker, OffloadingWorker)


def test_npu_worker_routes_store_and_load_by_direction():
    calls = []

    class _FakeHandler:
        def __init__(self, direction):
            self.direction = direction

        def transfer_async(self, job_id, src_spec, dst_spec):
            calls.append((self.direction, job_id, src_spec, dst_spec))
            return True

    worker = npu_mod.NPUOffloadingWorker.__new__(npu_mod.NPUOffloadingWorker)
    worker.npu_to_cpu_handler = _FakeHandler("store")
    worker.cpu_to_npu_handler = _FakeHandler("load")

    assert worker.submit_store(1, "npu", "cpu")
    assert worker.submit_load(2, "cpu", "npu")
    assert calls == [
        ("store", 1, "npu", "cpu"),
        ("load", 2, "cpu", "npu"),
    ]


def test_npu_worker_releases_handlers_before_mmap():
    calls = []

    class _FakeHandler:
        def __init__(self, name):
            self.name = name

        def shutdown(self):
            calls.append(self.name)

    class _FakeRegion:
        def cleanup(self):
            calls.append("mmap")

    worker = npu_mod.NPUOffloadingWorker.__new__(npu_mod.NPUOffloadingWorker)
    worker.npu_to_cpu_handler = _FakeHandler("store")
    worker.cpu_to_npu_handler = _FakeHandler("load")
    worker._mmap_region = _FakeRegion()

    worker.shutdown()

    assert calls == ["store", "load", "mmap"]
    assert worker._mmap_region is None


def test_npu_spec_caches_worker_without_upstream_platform_gate(monkeypatch):
    worker = object()
    spec = npu_mod.NPUOffloadingSpec.__new__(npu_mod.NPUOffloadingSpec)
    spec._worker = None
    create_calls = 0

    def create_worker(kv_caches):
        nonlocal create_calls
        create_calls += 1
        return worker

    monkeypatch.setattr(spec, "create_worker", create_worker)
    kv_caches = object()

    assert spec.get_worker(kv_caches) is worker
    assert spec.get_worker(kv_caches) is worker
    assert create_calls == 1


def test_tiering_create_worker_matches_shared_region_signature(monkeypatch):
    """create_worker must call SharedOffloadRegion with a valid signature."""
    captured: dict = {}

    class _FakeRegion:
        def __init__(self, **kwargs):
            # Raises TypeError if kwargs don't bind to the real upstream
            # __init__ (this is exactly what the original bug violated).
            inspect.signature(SharedOffloadRegion.__init__).bind(self, **kwargs)
            captured.update(kwargs)
            captured["region"] = self

    sentinel_worker = object()

    def _fake_worker(**kwargs):
        captured["worker_kwargs"] = kwargs
        return sentinel_worker

    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", _FakeRegion)
    monkeypatch.setattr(npu_mod, "NPUOffloadingWorker", _fake_worker)
    monkeypatch.setattr(
        npu_mod.torch,
        "npu",
        SimpleNamespace(current_device=lambda: 0),
        raising=False,
    )

    spec = npu_mod.NPUTieringOffloadingSpec.__new__(npu_mod.NPUTieringOffloadingSpec)
    spec.vllm_config = SimpleNamespace(
        instance_id="inst",
        parallel_config=SimpleNamespace(world_size=2),
    )
    spec.cpu_page_size_per_worker = 64
    spec.num_blocks = 10
    spec.block_size_factor = 1
    # Aligned per-block row stride exposed by CPUOffloadingSpec; the mmap region
    # derives its total size from num_blocks * this value.
    spec.kv_bytes_per_offloaded_block = 4096

    result = spec.create_worker(kv_caches=object())

    assert result is sentinel_worker
    # New upstream signature: size is derived from num_blocks * kv_bytes_per_block.
    assert captured["kv_bytes_per_block"] == 4096
    assert captured["num_blocks"] == 10
    assert captured["cpu_page_size"] == 64
    assert captured["rank"] == 0
    assert captured["instance_id"] == "inst"
    # The removed kwargs must never reappear.
    assert "total_size_bytes" not in captured
    assert "num_workers" not in captured
    # The NPU worker must receive the mmap region it will clean up.
    assert captured["worker_kwargs"]["mmap_region"] is captured["region"]


def test_tiering_create_worker_cleans_mmap_on_construction_failure(monkeypatch):
    cleaned = []

    class _FakeRegion:
        def __init__(self, **kwargs):
            pass

        def cleanup(self):
            cleaned.append(True)

    def _failing_worker(**kwargs):
        raise ValueError("invalid KV layout")

    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", _FakeRegion)
    monkeypatch.setattr(npu_mod, "NPUOffloadingWorker", _failing_worker)
    monkeypatch.setattr(
        npu_mod.torch,
        "npu",
        SimpleNamespace(current_device=lambda: 0),
        raising=False,
    )

    spec = npu_mod.NPUTieringOffloadingSpec.__new__(npu_mod.NPUTieringOffloadingSpec)
    spec.vllm_config = SimpleNamespace(instance_id="inst")
    spec.cpu_page_size_per_worker = 64
    spec.num_blocks = 10
    spec.block_size_factor = 1
    spec.kv_bytes_per_offloaded_block = 4096

    with pytest.raises(ValueError, match="invalid KV layout"):
        spec.create_worker(kv_caches=object())

    assert cleaned == [True]


def test_upstream_compatible_layout_uses_upstream_canonicalization(monkeypatch):
    layer_name = "model.layers.0.self_attn"
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )
    worker = NPUOffloadingConnectorWorker.__new__(NPUOffloadingConnectorWorker)
    worker.spec = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=4,
            kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec)],
        )
    )
    calls = []

    def _capture_upstream(self, kv_caches):
        calls.append(kv_caches)

    monkeypatch.setattr(
        OffloadingConnectorWorker,
        "register_kv_caches",
        _capture_upstream,
    )
    caches = {layer_name: torch.empty((4, 2, 1, 6), dtype=torch.bfloat16)}

    worker.register_kv_caches(caches)

    assert calls == [caches]


def test_split_attention_cache_is_canonicalized_without_copy():
    layer_name = "model.layers.0.self_attn"
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )
    worker = NPUOffloadingConnectorWorker.__new__(NPUOffloadingConnectorWorker)
    worker.spec = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=4,
            kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec)],
        )
    )
    captured = []
    worker._init_worker = captured.append
    key = torch.empty((4, 2, 3), dtype=torch.bfloat16)
    value = torch.empty((4, 2, 3), dtype=torch.bfloat16)

    worker.register_kv_caches({layer_name: (key, value)})

    canonical = captured[0]
    assert [tensor.tensor.shape for tensor in canonical.tensors] == [
        (4, 12),
        (4, 12),
    ]
    assert [ref.page_size_bytes for ref in canonical.group_data_refs[0]] == [
        12,
        12,
    ]
    assert canonical.tensors[0].tensor.data_ptr() == key.data_ptr()
    assert canonical.tensors[1].tensor.data_ptr() == value.data_ptr()


def test_mamba_states_with_independent_storage_are_canonicalized():
    layer_name = "model.layers.0.mixer"
    spec = MambaSpec(
        block_size=1,
        shapes=((2,), (3,)),
        dtypes=(torch.int8, torch.int8),
        page_size_padded=8,
    )
    worker = NPUOffloadingConnectorWorker.__new__(NPUOffloadingConnectorWorker)
    worker.spec = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=4,
            kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec)],
        )
    )
    captured = []
    worker._init_worker = captured.append
    first_state = torch.empty((4, 2), dtype=torch.int8)
    second_state = torch.empty((4, 3), dtype=torch.int8)

    worker.register_kv_caches({layer_name: [first_state, second_state]})

    canonical = captured[0]
    assert [tensor.tensor.shape for tensor in canonical.tensors] == [
        (4, 2),
        (4, 3),
    ]
    assert [ref.page_size_bytes for ref in canonical.group_data_refs[0]] == [2, 3]
    assert canonical.tensors[0].tensor.data_ptr() == first_state.data_ptr()
    assert canonical.tensors[1].tensor.data_ptr() == second_state.data_ptr()

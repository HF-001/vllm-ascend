import torch
from typing_extensions import override
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec as _CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec as _TieringOffloadingSpec

from vllm_ascend.kv_offload.cpu_npu import NPUOffloadingWorker


class _NPUWorkerMixin:
    _worker: NPUOffloadingWorker | None

    def create_worker(self, kv_caches: CanonicalKVCaches) -> NPUOffloadingWorker:
        raise NotImplementedError

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        # CPUOffloadingSpec intentionally rejects platforms other than
        # CUDA/XPU. Preserve its worker cache/lifecycle while replacing that
        # platform gate with the Ascend implementation.
        if self._worker is None:
            self._worker = self.create_worker(kv_caches)
        return self._worker


class NPUOffloadingSpec(_NPUWorkerMixin, _CPUOffloadingSpec):
    """Ascend NPU implementation of vLLM's CPU KV offloading spec."""

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> NPUOffloadingWorker:
        return NPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
        )


class NPUTieringOffloadingSpec(_NPUWorkerMixin, _TieringOffloadingSpec):
    """Ascend NPU implementation of vLLM's multi-tier KV offloading spec."""

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> NPUOffloadingWorker:
        # Mirror upstream TieringOffloadingSpec.create_worker, but route the
        # worker-side transfers through the Ascend worker and resolve the
        # worker slot via the NPU device index.
        #
        # SharedOffloadRegion now sizes the mmap internally from
        # ``num_blocks * kv_bytes_per_block`` (the aligned per-block row stride
        # exposed as ``kv_bytes_per_offloaded_block``); the old
        # ``total_size_bytes`` / ``num_workers`` kwargs were removed upstream.
        rank = torch.npu.current_device()
        worker_mmap = SharedOffloadRegion(
            instance_id=self.vllm_config.instance_id,
            num_blocks=self.num_blocks,
            rank=rank,
            kv_bytes_per_block=self.kv_bytes_per_offloaded_block,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        try:
            return NPUOffloadingWorker(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
                mmap_region=worker_mmap,
            )
        except Exception:
            # The worker normally owns the region, but construction can fail
            # before ownership is transferred (for example on an unsupported
            # KV layout). Do not leave a shared-memory file/mapping behind.
            worker_mmap.cleanup()
            raise

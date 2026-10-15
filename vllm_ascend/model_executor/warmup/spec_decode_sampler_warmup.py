# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm MRV2 speculative sampling without changing live request/RNG state."""

from typing import TYPE_CHECKING

import torch

from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p_npu
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

_MAX_WARMUP_REQS = 2
_WARMUP_TOP_K = 10
_WARMUP_TOP_P = 0.9


@torch.inference_mode()
def spec_decode_sampler_warmup(worker: "NPUWorker") -> None:
    """Exercise native greedy logits and FP32 processed logits before capture.

    MRV1 uses different rejection kernels. MRV2 top-k/top-p is implemented
    with PyTorch NPU ops, so registering CUDA Triton warmups does not help.
    """
    if not worker.use_v2_model_runner or not worker.vllm_config.kernel_config.enable_jit_warmup:
        return
    runner = worker.model_runner
    if not runner.is_last_pp_rank or runner.rejection_sampler is None:
        return

    num_steps = runner.num_speculative_steps
    if num_steps <= 0:
        return
    device = worker.device
    vocab_size = runner.vocab_size
    logits_dtype = worker.model_config.head_dtype
    cached_draft_logits = runner.speculator.draft_logits
    synthetic_rates = runner.rejection_sampler.synthetic_conditional_rates

    # Match vocabulary widths and draft dtype without touching the persistent
    # logits cache. One and two requests cover singleton and batched launches.
    for num_reqs in sorted({1, min(_MAX_WARMUP_REQS, runner.max_num_reqs)}):
        num_logits = num_reqs * (num_steps + 1)
        cu_num_logits = torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * (num_steps + 1)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)
        expanded_idx_mapping = idx_mapping.repeat_interleave(num_steps + 1)
        local_pos = torch.arange(num_steps + 1, dtype=torch.int32, device=device).repeat(num_reqs)
        positions = local_pos.to(torch.int64)
        draft_sampled = torch.zeros(num_logits, dtype=torch.int32, device=device)
        seeds = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        draft_logits = (
            torch.zeros(
                (num_reqs, num_steps, cached_draft_logits.shape[-1]),
                dtype=cached_draft_logits.dtype,
                device=device,
            )
            if cached_draft_logits is not None
            else None
        )

        for dtype, temp in ((logits_dtype, 0.0), (torch.float32, 1.0)):
            logits = torch.zeros((num_logits, vocab_size), dtype=dtype, device=device)
            temperature = torch.full((num_reqs,), temp, dtype=torch.float32, device=device)
            rejection_sample(
                logits,
                draft_logits,
                draft_sampled,
                cu_num_logits,
                positions,
                idx_mapping,
                expanded_idx_mapping,
                local_pos,
                temperature,
                seeds,
                num_steps,
                synthetic_conditional_rates=synthetic_rates,
            )

        top_k = torch.full((num_logits,), min(_WARMUP_TOP_K, vocab_size), dtype=torch.int32, device=device)
        top_p = torch.full((num_logits,), _WARMUP_TOP_P, dtype=torch.float32, device=device)
        for k, p in ((top_k, None), (None, top_p), (top_k, top_p)):
            logits = torch.zeros((num_logits, vocab_size), dtype=torch.float32, device=device)
            apply_top_k_top_p_npu(logits, k, p)

    # Finish temporary-buffer work before graph capture and memory accounting.
    torch.npu.synchronize()

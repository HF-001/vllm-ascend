# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRV2 warmup must cover serving tensor signatures without changing state."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.model_executor.warmup import spec_decode_sampler_warmup as sw


def _worker(dtype=torch.bfloat16, with_draft=False, max_num_reqs=4):
    draft_logits = torch.ones(4, 3, 8193, dtype=torch.float16) if with_draft else None
    return SimpleNamespace(
        use_v2_model_runner=True,
        vllm_config=SimpleNamespace(kernel_config=SimpleNamespace(enable_jit_warmup=True)),
        model_config=SimpleNamespace(head_dtype=dtype),
        device=torch.device("cpu"),
        model_runner=SimpleNamespace(
            is_last_pp_rank=True,
            num_speculative_steps=3,
            max_num_reqs=max_num_reqs,
            vocab_size=8193,
            speculator=SimpleNamespace(draft_logits=draft_logits),
            rejection_sampler=SimpleNamespace(synthetic_conditional_rates=None),
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_draft", [False, True])
@pytest.mark.parametrize("max_num_reqs", [1, 4])
def test_warmup_matches_live_tensor_signatures(dtype, with_draft, max_num_reqs):
    worker = _worker(dtype, with_draft, max_num_reqs)
    original_rng = torch.get_rng_state().clone()
    cached = worker.model_runner.speculator.draft_logits
    original_cache = cached.clone() if cached is not None else None
    with (
        patch.object(sw, "rejection_sample") as reject,
        patch.object(sw, "apply_top_k_top_p_npu") as top_k_top_p,
        patch.object(torch.npu, "synchronize") as synchronize,
    ):
        sw.spec_decode_sampler_warmup(worker)

    batches = [1] if max_num_reqs == 1 else [1, 2]
    assert reject.call_count == 2 * len(batches)
    for call, (batch, target_dtype, temperature_value) in zip(
        reject.call_args_list,
        [(b, d, t) for b in batches for d, t in ((dtype, 0.0), (torch.float32, 1.0))],
        strict=True,
    ):
        logits, draft, tokens, cu, positions, mapping, expanded, local, temperature, seeds, steps = call.args
        assert logits.shape == (batch * 4, 8193)
        assert logits.dtype == target_dtype
        assert tokens.dtype == torch.int32
        assert positions.dtype == seeds.dtype == mapping.dtype == expanded.dtype == torch.int64
        assert cu.dtype == local.dtype == torch.int32
        assert cu.tolist() == [i * 4 for i in range(batch + 1)]
        assert mapping.tolist() == list(range(batch))
        assert expanded.tolist() == [i for i in range(batch) for _ in range(4)]
        assert local.tolist() == list(range(4)) * batch
        assert temperature.tolist() == [temperature_value] * batch
        assert steps == 3
        assert call.kwargs["synthetic_conditional_rates"] is None
        if with_draft:
            assert draft.shape == (batch, 3, 8193)
            assert draft.dtype == cached.dtype
            assert draft.data_ptr() != cached.data_ptr()
        else:
            assert draft is None

    assert top_k_top_p.call_count == 3 * len(batches)
    for index, call in enumerate(top_k_top_p.call_args_list):
        logits, k, p = call.args
        assert logits.dtype == torch.float32
        assert (k is not None, p is not None) == [(True, False), (False, True), (True, True)][index % 3]
    synchronize.assert_called_once()
    assert torch.equal(torch.get_rng_state(), original_rng)
    if cached is not None:
        assert torch.equal(cached, original_cache)


@pytest.mark.parametrize("skip", ["mrv1", "disabled", "non_last_pp", "no_rejection", "no_steps"])
def test_warmup_skips_inapplicable_workers(skip):
    worker = _worker()
    if skip == "mrv1":
        worker.use_v2_model_runner = False
    elif skip == "disabled":
        worker.vllm_config.kernel_config.enable_jit_warmup = False
    elif skip == "non_last_pp":
        worker.model_runner.is_last_pp_rank = False
    elif skip == "no_rejection":
        worker.model_runner.rejection_sampler = None
    else:
        worker.model_runner.num_speculative_steps = 0
    with (
        patch.object(sw, "rejection_sample") as reject,
        patch.object(sw, "apply_top_k_top_p_npu") as top_k_top_p,
        patch.object(torch.npu, "synchronize") as synchronize,
    ):
        sw.spec_decode_sampler_warmup(worker)
    reject.assert_not_called()
    top_k_top_p.assert_not_called()
    synchronize.assert_not_called()


def test_warmup_preserves_synthetic_rates():
    worker = _worker(max_num_reqs=1)
    rates = torch.tensor([0.8, 0.5, 0.2])
    worker.model_runner.rejection_sampler.synthetic_conditional_rates = rates
    with (
        patch.object(sw, "rejection_sample") as reject,
        patch.object(sw, "apply_top_k_top_p_npu"),
        patch.object(torch.npu, "synchronize"),
    ):
        sw.spec_decode_sampler_warmup(worker)
    assert all(call.kwargs["synthetic_conditional_rates"] is rates for call in reject.call_args_list)
    assert rates.tolist() == pytest.approx([0.8, 0.5, 0.2])

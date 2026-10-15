# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``kernel_warmup``."""

import importlib
from unittest.mock import patch

from tests.ut.model_executor.warmup.helpers import make_mock_worker

kw = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")


@patch.object(kw, "logger")
@patch.object(kw, "triton_rms_warmup")
@patch.object(kw, "penalties_triton_warmup")
@patch.object(kw, "rejection_sampler_triton_warmup")
@patch.object(kw, "indexer_triton_warmup")
@patch.object(kw, "HAS_TRITON", True)
def test_kernel_warmup(mock_indexer, mock_rej, mock_pen, mock_rms, mock_logger):
    worker = make_mock_worker()
    worker.use_v2_model_runner = False
    kw.kernel_warmup(worker)

    mock_rej.assert_called_once_with(worker)
    mock_pen.assert_called_once_with(worker)
    mock_rms.assert_called_once_with(worker)
    mock_indexer.assert_called_once_with(worker)


def test_kernel_warmup_includes_mrv2_sampling():
    worker = make_mock_worker()
    worker.use_v2_model_runner = True
    with (
        patch.object(kw, "HAS_TRITON", True),
        patch.object(kw, "_run_warmup") as run,
        patch(
            "vllm_ascend.model_executor.warmup.spec_decode_sampler_warmup.spec_decode_sampler_warmup"
        ) as sampling_warmup,
    ):
        kw.kernel_warmup(worker)
    run.assert_any_call("spec_decode_sampler", sampling_warmup, worker)
    run.assert_any_call("rejection_sampler", kw.rejection_sampler_triton_warmup, worker)

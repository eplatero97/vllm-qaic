# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""End-to-end tests for ngram/suffix speculative decoding in PYT (eager) mode.

Requires QAIC hardware and a PYT-mode environment (torch_qaic installed).
Modeled on ``examples/qaic_ngram_suffix.py``.

Correctness strategy: greedy rejection sampling is mathematically exact, so
speculative-decode output MUST equal the non-speculative greedy baseline
token-for-token.  Each SpD test compares its per-request ``token_ids`` against
a baseline built once (no speculative_config) on the same prompts and sampling
params.  This is a strictly stronger check than the reference smoke tests
(which only assert output is non-empty) and directly validates the rejection-
sampler shim end-to-end.

Run::

    .venv_eager/bin/python -m pytest -s \
        tests/test_qaic/spec_decode/e2e/test_ngram_suffix_integration.py \
        --test-device-group '[0]' -v
"""

import gc
import os

import pytest

from vllm import LLM, SamplingParams
from vllm_qaic.platform_base import QaicPlatform

# Skip the whole module in AOT mode — this validates the PYT-mode shim.
pytestmark = pytest.mark.skipif(
    QaicPlatform.is_aot,
    reason="PYT-mode SpD E2E test; requires torch_qaic (eager mode).",
)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Repetition biases the ngram/suffix proposer toward producing (and the target
# toward accepting) draft tokens, exercising the accept path in the shim.
PROMPTS = [
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
    "My name is",
]

CTX_LEN = 256
SEQ_LEN = 128
DECODE_BSZ = 16
NUM_SPEC_TOKENS = 5

# Deterministic greedy sampling so baseline and SpD are directly comparable.
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=32, seed=42)


def _base_llm_kwargs() -> dict:
    """LLM kwargs shared by baseline and SpD runs.

    Mirrors examples/qaic_eager_mode.py (the PYT/eager reference), NOT
    qaic_ngram_suffix.py: mxfp6/mxint8 are AOT-only concepts (QaicQuantConfig
    returns no quant method, which eager-mode linear layers reject), so PYT
    mode runs the model in native fp16 with enforce_eager=True.
    """
    return dict(
        model=MODEL,
        max_num_seqs=DECODE_BSZ,
        max_model_len=CTX_LEN,
        long_prefill_token_threshold=SEQ_LEN,
        tensor_parallel_size=1,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        # 1.0 demands 100% of device memory and fails the worker free-memory
        # pre-flight; 0.9 leaves headroom (matches qaic_eager_mode.py).
        gpu_memory_utilization=0.9,
    )


def _token_ids(outputs) -> list[list[int]]:
    """Extract per-request generated token ids in prompt order."""
    return [list(o.outputs[0].token_ids) for o in outputs]


@pytest.fixture(scope="session", autouse=True)
def _qaic_visible_devices(device_group):
    """Export QAIC_VISIBLE_DEVICES from the requested device group.

    QaicPlatform.check_and_update_config only reads device_group into
    os.environ when QAIC_VISIBLE_DEVICES is unset, and that assignment
    expects a string. Setting it here — the way examples/qaic_eager_mode.py
    does — keeps device selection correct without additional_config.
    """
    key = QaicPlatform.device_control_env_var  # "QAIC_VISIBLE_DEVICES"
    prev = os.environ.get(key)
    os.environ[key] = ",".join(str(q) for q in device_group)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


@pytest.fixture(scope="session")
def baseline_token_ids() -> list[list[int]]:
    """Greedy generation with NO speculative decoding — the ground truth.

    Device selection is handled by the autouse _qaic_visible_devices fixture
    (QAIC_VISIBLE_DEVICES), so no additional_config is needed.
    """
    llm = LLM(**_base_llm_kwargs())
    try:
        outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
        return _token_ids(outputs)
    finally:
        del llm
        gc.collect()


def _run_spd(method: str) -> list[list[int]]:
    llm = LLM(
        **_base_llm_kwargs(),
        speculative_config={
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "method": method,
        },
    )
    try:
        outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
        return _token_ids(outputs)
    finally:
        del llm
        gc.collect()


def test_ngram_matches_baseline(baseline_token_ids):
    spd_token_ids = _run_spd("ngram")

    assert len(spd_token_ids) == len(baseline_token_ids)
    for i, (spd, base) in enumerate(
        zip(spd_token_ids, baseline_token_ids, strict=False)
    ):
        assert len(base) > 0, f"baseline request {i} produced no tokens"
        assert spd == base, (
            f"ngram SpD output diverged from greedy baseline for prompt {i}:\n"
            f"  baseline={base}\n  spd     ={spd}"
        )


def test_suffix_matches_baseline(baseline_token_ids):
    spd_token_ids = _run_spd("suffix")

    assert len(spd_token_ids) == len(baseline_token_ids)
    for i, (spd, base) in enumerate(
        zip(spd_token_ids, baseline_token_ids, strict=False)
    ):
        assert len(base) > 0, f"baseline request {i} produced no tokens"
        assert spd == base, (
            f"suffix SpD output diverged from greedy baseline for prompt {i}:\n"
            f"  baseline={base}\n  spd     ={spd}"
        )

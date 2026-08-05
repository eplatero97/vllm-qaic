# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Async-vs-sync draft_model SpD correctness harness UNDER REQUEST TURNOVER.

Sibling of test_async_spec_draft_model_aot.py. That test fills every decode
slot once (4 prompts, no turnover), so it never exercises the async path when
the batch composition CHANGES between steps -- i.e. when requests finish and
new ones are admitted into freed slots while other requests are mid-generation.

Turnover is where the async draft_model plumbing is most at risk: the async
path defers proposal into get_output() (one step lagged), and per-request state
is keyed a mix of ways (positional accepted-count list, req_id->index map,
QaicDraftModelProposer.propose()'s purely index-keyed batch_indices/positions).
If _update_states condenses/reorders the batch between the deferred proposal and
the next step, a positional read could land on the wrong request.

Under GREEDY sampling, speculative decoding is LOSSLESS: sync-SpD, async-SpD,
and non-spec greedy must all emit token-for-token identical output regardless of
draft acceptance rate. So this test decides one question cleanly:

  * async matches baseline under turnover  -> async plumbing is CORRECT; any
    throughput regression is a "correct-but-slower" acceptance-rate property.
  * async diverges                          -> real alignment bug in the async
    per-request state keying.

Turnover is forced two ways at once: (1) a pool of distinct, varied-length
prompts, and (2) per-prompt varied max_tokens, so requests finish at staggered
steps and new ones are admitted mid-flight -- with a small max_num_seqs so
len(pool) >> slots. Each request is compared to its OWN sync non-spec baseline
generated with the identical per-prompt sampling params.

Default model pair is TinyLlama-as-both (same as the sibling test; satisfies
verify_equal_vocab_size_if_draft_model() without a gated pair, keeps the suite
runnable everywhere). Set VLLM_QAIC_TEST_REALISTIC_PAIR=1 to swap in the
Meta-Llama-3-8B-Instruct / Llama-3.2-1B-Instruct pair (shared 128256 vocab),
which is the pairing that actually exhibits the acceptance-rate collapse.

    # pytest, TinyLlama-as-both (default), single QID
    sg qaic -c '.venv/bin/python -m pytest -s \
        tests/test_qaic/spec_decode/e2e/test_async_spec_draft_model_turnover_aot.py \
        --test-device-group "[0]" -v'

    # pytest, realistic Llama-3 pair (gated models must be available locally)
    sg qaic -c 'VLLM_QAIC_TEST_REALISTIC_PAIR=1 .venv/bin/python -m pytest -s \
        tests/test_qaic/spec_decode/e2e/test_async_spec_draft_model_turnover_aot.py \
        --test-device-group "[0]" -v'
"""

import gc
import os

import pytest

from vllm import LLM, SamplingParams
from vllm_qaic.platform_base import QaicPlatform

# This validates the AOT path; skip in PYT/eager mode.
pytestmark = pytest.mark.skipif(
    not QaicPlatform.is_aot,
    reason="AOT-mode async+SpD E2E test; requires AOT (no torch_qaic).",
)

_REALISTIC = os.environ.get("VLLM_QAIC_TEST_REALISTIC_PAIR", "0") == "1"

if _REALISTIC:
    # Realistic target->smaller-draft pairing. Shares vocab 128256, so it
    # satisfies verify_equal_vocab_size_if_draft_model(). This is the pair that
    # exhibits the async acceptance-rate collapse under turnover.
    TARGET_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    CTX_LEN = 2048
else:
    # Mechanism check: TinyLlama as BOTH target and draft (no gated models).
    TARGET_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DRAFT_MODEL = TARGET_MODEL
    CTX_LEN = 512

SEQ_LEN = 128
NUM_SPEC_TOKENS = 3

# Small batch so len(PROMPTS) >> slots -> requests must cycle through slots.
MAX_NUM_SEQS = 2

# Distinct, varied-length prompts. Combined with the varied per-prompt
# max_tokens below, requests finish at staggered steps, forcing new requests to
# be admitted into freed slots while others are still generating.
PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "Once upon a time, in a land far far away, there lived a wise old",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
    "The quick brown fox jumps over the lazy",
    "Photosynthesis is the process by which plants convert sunlight into",
    "In the beginning God created the heavens and the",
    "To be, or not to be, that is the",
    "The mitochondria is the powerhouse of the",
    "Roses are red, violets are blue, sugar is sweet and so are",
    "The three primary colors are red, blue, and",
    "E equals m c squared is the famous equation by Albert",
]

# Per-prompt max_tokens, cycled so completions stagger across decode steps.
_MAX_TOKENS_CYCLE = [16, 40, 24, 56, 32, 48]

# Deterministic greedy so baseline and SpD are directly comparable, per prompt.
SAMPLING_PARAMS = [
    SamplingParams(
        temperature=0.0,
        max_tokens=_MAX_TOKENS_CYCLE[i % len(_MAX_TOKENS_CYCLE)],
        seed=42,
    )
    for i in range(len(PROMPTS))
]


def _base_llm_kwargs(
    async_scheduling: bool,
    max_num_seqs: int = MAX_NUM_SEQS,
    qid: int = 0,
    target_cores: int = 10,
    draft_cores: int = 6,
) -> dict:
    return dict(
        model=TARGET_MODEL,
        max_num_seqs=max_num_seqs,
        max_model_len=CTX_LEN,
        long_prefill_token_threshold=SEQ_LEN,
        tensor_parallel_size=1,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        async_scheduling=async_scheduling,
        enable_prefix_caching=False,
        gpu_memory_utilization=1.0,
        disable_log_stats=False,
        additional_config={
            "override_qaic_config": {
                "device_group": [qid],
                "num_cores": target_cores,
            },
            "draft_override_qaic_config": {
                "device_group": [qid],  # same device as the target
                "num_cores": draft_cores,
            },
        },
    )


def _token_ids(outputs) -> list[list[int]]:
    return [list(o.outputs[0].token_ids) for o in outputs]


def _run(use_spec: bool, async_scheduling: bool) -> list[list[int]]:
    kwargs = _base_llm_kwargs(async_scheduling)
    if use_spec:
        kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": NUM_SPEC_TOKENS,
        }
    llm = LLM(**kwargs)
    try:
        outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
        return _token_ids(outputs)
    finally:
        del llm
        gc.collect()


@pytest.fixture(scope="session", autouse=True)
def _qaic_visible_devices(device_group):
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
    """Greedy, NO spec decode, sync scheduling, run over the SAME turnover pool
    at the same small max_num_seqs -- so the ground-truth oracle itself
    exercises request turnover."""
    return _run(use_spec=False, async_scheduling=False)


def _assert_match(spd, base, label):
    assert len(spd) == len(base)
    for i, (s, b) in enumerate(zip(spd, base, strict=False)):
        assert len(b) > 0, f"baseline request {i} produced no tokens"
        assert s == b, (
            f"{label} SpD output diverged from greedy baseline for prompt {i} "
            f"under turnover:\n  prompt   ={PROMPTS[i]!r}\n"
            f"  baseline ={b}\n  spd      ={s}"
        )


def test_sync_draft_model_turnover_matches_baseline(baseline_token_ids):
    """Oracle: sync draft_model SpD must match the non-spec greedy baseline
    under turnover, before async is judged at all."""
    spd = _run(use_spec=True, async_scheduling=False)
    _assert_match(spd, baseline_token_ids, "sync draft_model (turnover)")


def test_async_draft_model_turnover_matches_baseline(baseline_token_ids):
    """The feature under test: async draft_model SpD must be token-for-token
    identical to the greedy baseline under request turnover."""
    spd = _run(use_spec=True, async_scheduling=True)
    _assert_match(spd, baseline_token_ids, "async draft_model (turnover)")


def test_async_matches_sync_draft_model_turnover():
    """Localizer: async-SpD vs sync-SpD directly. If the baseline tests fail
    together, this isolates whether the divergence is async-specific (this
    fails) or a target/draft checkpoint mismatch shared by both (this passes)."""
    sync_spd = _run(use_spec=True, async_scheduling=False)
    async_spd = _run(use_spec=True, async_scheduling=True)
    _assert_match(async_spd, sync_spd, "async-vs-sync draft_model (turnover)")

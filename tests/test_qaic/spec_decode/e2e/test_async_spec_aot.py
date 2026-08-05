# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Async-vs-sync speculative-decoding harness for QAIC AOT mode.

This is both a correctness oracle and a perf comparison for enabling
``async_scheduling=True`` together with ngram/suffix speculative decoding on
QAIC AOT.

Correctness strategy: greedy rejection sampling is mathematically exact, so
SpD output MUST equal the non-speculative greedy baseline token-for-token, in
BOTH sync and async scheduling modes. Async and sync SpD must therefore also
match each other.

Runnable both as a pytest module and as a standalone script:

    # pytest (single QID)
    sg qaic -c '.venv/bin/python -m pytest -s \
        tests/test_qaic/spec_decode/e2e/test_async_spec_aot.py \
        --test-device-group "[0]" -v'

    # standalone perf comparison (prints throughput table)
    sg qaic -c 'QAIC_VISIBLE_DEVICES=0 .venv/bin/python \
        tests/test_qaic/spec_decode/e2e/test_async_spec_aot.py'
"""

import gc
import os
import time

import pytest

from vllm import LLM, SamplingParams
from vllm_qaic.platform_base import QaicPlatform

# This validates the AOT path; skip in PYT/eager mode.
pytestmark = pytest.mark.skipif(
    not QaicPlatform.is_aot,
    reason="AOT-mode async+SpD E2E test; requires AOT (no torch_qaic).",
)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Repetition biases the ngram/suffix proposer toward producing (and the target
# toward accepting) draft tokens, exercising the accept path.
PROMPTS = [
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
    "My name is",
    "Once upon a time, in a land far far away, there lived a",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19",
]

CTX_LEN = 256
SEQ_LEN = 128
DECODE_BSZ = 16
NUM_SPEC_TOKENS = 5

# Deterministic greedy sampling so baseline and SpD are directly comparable.
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=64, seed=42)


def _base_llm_kwargs(async_scheduling: bool, max_num_seqs: int = DECODE_BSZ) -> dict:
    return dict(
        model=MODEL,
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
    )


def _token_ids(outputs) -> list[list[int]]:
    return [list(o.outputs[0].token_ids) for o in outputs]


def _run(method: str | None, async_scheduling: bool) -> list[list[int]]:
    kwargs = _base_llm_kwargs(async_scheduling)
    if method is not None:
        kwargs["speculative_config"] = {
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "method": method,
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
    """Greedy generation, NO speculative decoding, sync scheduling — ground truth."""
    return _run(method=None, async_scheduling=False)


@pytest.mark.parametrize("method", ["ngram", "suffix"])
def test_sync_spd_matches_baseline(method, baseline_token_ids):
    """Step 0 oracle: sync SpD must match the non-spec greedy baseline."""
    spd = _run(method=method, async_scheduling=False)
    _assert_match(spd, baseline_token_ids, f"sync {method}")


@pytest.mark.parametrize("method", ["ngram", "suffix"])
def test_async_spd_matches_baseline(method, baseline_token_ids):
    """The feature under test: async SpD must match the greedy baseline."""
    spd = _run(method=method, async_scheduling=True)
    _assert_match(spd, baseline_token_ids, f"async {method}")


def _assert_match(spd, base, label):
    assert len(spd) == len(base)
    for i, (s, b) in enumerate(zip(spd, base, strict=False)):
        assert len(b) > 0, f"baseline request {i} produced no tokens"
        assert s == b, (
            f"{label} SpD output diverged from greedy baseline for prompt {i}:\n"
            f"  baseline={b}\n  spd     ={s}"
        )


def _perf_main():
    """Standalone throughput comparison: async vs sync ngram SpD.

    Sweeps max_num_seqs; async scheduling should widen its lead as batch size
    grows because it hides input preparation behind on-device inference.
    ngram only (suffix needs arctic-inference).
    """
    method = "ngram"
    # A batch of repetitive prompts drives high ngram acceptance and keeps the
    # decode batch full so async overlap has something to hide behind.
    prompt_pool = [
        "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
        "Once upon a time, once upon a time, once upon a time, once upon a",
    ]
    results = {}
    for mns in (1, 4, 8):
        total_prompts = [prompt_pool[i % len(prompt_pool)] for i in range(mns * 4)]
        for async_sched in (False, True):
            kwargs = _base_llm_kwargs(async_sched, max_num_seqs=mns)
            kwargs["speculative_config"] = {
                "num_speculative_tokens": NUM_SPEC_TOKENS,
                "method": method,
            }
            llm = LLM(**kwargs)
            try:
                llm.generate(prompt_pool, SAMPLING_PARAMS)  # warmup
                t0 = time.perf_counter()
                outputs = llm.generate(total_prompts, SAMPLING_PARAMS)
                dt = time.perf_counter() - t0
            finally:
                del llm
                gc.collect()
            n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
            results[(mns, async_sched)] = n_out / dt
            print(
                f"[mns={mns} async={async_sched!s:5s}] "
                f"{n_out} tok in {dt:.2f}s = {n_out / dt:.1f} tok/s"
            )

    print("\n=== async vs sync ngram throughput (tok/s) ===")
    print(f"{'max_num_seqs':>12} {'sync':>10} {'async':>10} {'Δ%':>8}")
    for mns in (1, 4, 8):
        sync_tps = results[(mns, False)]
        async_tps = results[(mns, True)]
        delta = (async_tps - sync_tps) / sync_tps * 100.0
        print(f"{mns:>12} {sync_tps:>10.1f} {async_tps:>10.1f} {delta:>+7.1f}%")


if __name__ == "__main__":
    _perf_main()

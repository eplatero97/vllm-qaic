# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.envs import (
    environment_variables,
    maybe_convert_bool,
    maybe_convert_int,
)


if TYPE_CHECKING:
    VLLM_QAIC_COMPILER_ARGS: str | None = None
    VLLM_QAIC_DFS_EN: bool = True
    VLLM_QAIC_ASYNC_SPEC_SAMPLER_SKIP: bool = False
    VLLM_QAIC_MAX_CPU_THREADS: int | None = None
    VLLM_QAIC_MOS: int | None = None
    VLLM_QAIC_NUM_CORES: int | None = None
    VLLM_QAIC_QPC_PATH: str | None = None
    VLLM_QAIC_SPEC_ACCEPT_TRACE: bool = False
    VLLM_QAIC_STEP_TIMING: bool = False
    VLLM_TORCH_QAIC_PROFILER_DIR: str | None = None

# --8<-- [start:env-vars-definition]
qaic_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_QAIC_COMPILER_ARGS": lambda: os.getenv("VLLM_QAIC_COMPILER_ARGS", None),
    "VLLM_QAIC_DFS_EN": lambda: maybe_convert_bool(os.getenv("VLLM_QAIC_DFS_EN", None)),
    # Skip the rejection sampler + full-K SpD kernel on async decode steps where
    # the batch-wide real draft count is 0 (ngram/suffix only). EXPERIMENTAL and
    # OFF by default: the async + active_k=0 path has a known correctness issue
    # (output divergence) — see docs/qaic/async_scheduling_spec_decode_findings.md.
    # Set to 1 to opt in for perf experiments.
    "VLLM_QAIC_ASYNC_SPEC_SAMPLER_SKIP": lambda: maybe_convert_bool(
        os.getenv("VLLM_QAIC_ASYNC_SPEC_SAMPLER_SKIP", "0")
    ),
    "VLLM_QAIC_MAX_CPU_THREADS": lambda: maybe_convert_int(
        os.getenv("VLLM_QAIC_MAX_CPU_THREADS", None)
    ),
    "VLLM_QAIC_MOS": lambda: maybe_convert_int(os.getenv("VLLM_QAIC_MOS", None)),
    "VLLM_QAIC_NUM_CORES": lambda: maybe_convert_int(
        os.getenv("VLLM_QAIC_NUM_CORES", None)
    ),
    "VLLM_QAIC_QPC_PATH": lambda: os.getenv("VLLM_QAIC_QPC_PATH", None),
    # Emit a per-decode-step, per-request trace of the RAW proposed draft token
    # ids + accepted tokens on both sync and async paths, for offline
    # decomposition of the async spec-decode throughput gap (active-only
    # per-position acceptance + cross-run raw-draft diff). OFF by default; zero
    # overhead when off. See docs/qaic/async_scheduling_spec_decode_findings.md.
    "VLLM_QAIC_SPEC_ACCEPT_TRACE": lambda: maybe_convert_bool(
        os.getenv("VLLM_QAIC_SPEC_ACCEPT_TRACE", None)
    ),
    "VLLM_QAIC_STEP_TIMING": lambda: maybe_convert_bool(
        os.getenv("VLLM_QAIC_STEP_TIMING", None)
    ),
    "VLLM_TORCH_QAIC_PROFILER_DIR": lambda: os.getenv(
        "VLLM_TORCH_QAIC_PROFILER_DIR", None
    ),
}
environment_variables.update(qaic_environment_variables)
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    """
    Gets environment variables lazily.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())

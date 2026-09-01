# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Eager native-MTP speculative-decoding correctness test.

Run with visible subprocess output::

    .venv_eager/bin/python -m pytest -s -v \
        tests/test_qaic/spec_decode/e2e/test_mtp_integration.py \
        --test-device-group '[8,9]'

Override the default MiMo checkpoint with ``QAIC_EAGER_MTP_MODEL`` and its
tensor-parallel size with ``QAIC_EAGER_MTP_TENSOR_PARALLEL_SIZE``.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from vllm_qaic.platform_base import QaicPlatform

pytestmark = pytest.mark.skipif(
    QaicPlatform.is_aot,
    reason="Requires torch_qaic and eager PYT mode.",
)

_RUNNER_SCRIPT = Path(__file__).with_name("_run_mtp_generation.py")
_MODEL = os.environ.get("QAIC_EAGER_MTP_MODEL", "XiaomiMiMo/MiMo-7B-Base")


def _run_generation(mode: str, qid: int) -> dict:
    env = os.environ.copy()
    env[QaicPlatform.device_control_env_var] = str(qid)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "generation.json"
        result = subprocess.run(
            [
                sys.executable,
                str(_RUNNER_SCRIPT),
                mode,
                str(output_path),
                "--model",
                _MODEL,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            pytest.fail(
                f"child generation ({mode}, qid={qid}) failed with exit "
                f"{result.returncode}.\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
        return json.loads(output_path.read_text())


def test_eager_mtp_matches_greedy_baseline(device_group):
    if len(device_group) < 2:
        pytest.fail(
            "Eager MTP E2E requires two ready single-QID devices: "
            "baseline and speculative decoding."
        )

    baseline_result = _run_generation("baseline", device_group[0])
    speculative_result = _run_generation("mtp", device_group[1])

    assert baseline_result["mode"] == "baseline"
    assert speculative_result["mode"] == "mtp"
    assert speculative_result["model"] == _MODEL
    assert (
        speculative_result["spec_decode_metrics"]["vllm:spec_decode_num_draft_tokens"]
        > 0
    )
    assert (
        speculative_result["spec_decode_metrics"][
            "vllm:spec_decode_num_accepted_tokens"
        ]
        > 0
    )
    baseline_token_ids = baseline_result["token_ids"]
    speculative_token_ids = speculative_result["token_ids"]
    assert len(speculative_token_ids) == len(baseline_token_ids)
    for request_index, (speculative, baseline) in enumerate(
        zip(speculative_token_ids, baseline_token_ids, strict=True)
    ):
        assert baseline, f"baseline request {request_index} produced no tokens"
        assert speculative == baseline, (
            f"MTP SpD diverged for request {request_index}:\n"
            f"baseline={baseline}\n"
            f"speculative={speculative}"
        )

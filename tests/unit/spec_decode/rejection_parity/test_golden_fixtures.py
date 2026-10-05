# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Tier B: Numba rejection-sampler kernels vs recorded triton-cpu outputs.

No triton needed.  ``fixtures/rs_golden_v1.pt`` holds inputs and the real
upstream Triton output for each case (see ``tools/rs_parity/gen_golden_fixtures.py``).
Each kernel case is replayed through:

* ``prod`` -- the wrapper objects ``rejection_sampler_numba.install()`` puts in
  ``vllm.v1.sample.rejection_sampler`` (called as ``kernel[grid](*args, **kw)``);
* ``raw``  -- the raw ``numba_rejection_kernels`` functions.

``e2e:*`` cases call upstream ``rejection_sample()`` with the Numba backend
installed and recorded random draws.  Outputs must be ``torch.equal``; inputs
must not be mutated.  The fixture file and every case are sha256-checked against
``rs_golden_v1.json``.

    .venv_aot/bin/python -m pytest -s \
        tests/unit/spec_decode/rejection_parity/test_golden_fixtures.py
"""

from __future__ import annotations

import json
import os

import _parity_common as common
import pytest
import torch

if not os.path.exists(common.GOLDEN_PT):
    pytest.skip(f"golden fixture missing: {common.GOLDEN_PT}", allow_module_level=True)

GOLDEN = common.load_golden()
with open(common.golden_json_path(common.GOLDEN_PT)) as _f:
    META = json.load(_f)
CASES = GOLDEN["cases"]
KERNEL_CASES = sorted(n for n, c in CASES.items() if c["kind"] == "kernel")
E2E_CASES = sorted(n for n, c in CASES.items() if c["kind"] == "rejection_sample")


@pytest.fixture(scope="module")
def wrappers():
    w = common.production_wrappers()
    if w is None:
        pytest.skip(f"{common.PROD_MODULE} not importable")
    return w


@pytest.fixture
def numba_installed():
    from vllm_qaic.v1.sample import rejection_sampler_numba as rsn

    was = rsn._installed
    assert rsn.install() == "numba"
    try:
        yield
    finally:
        if not was:
            rsn.uninstall()


def test_fixture_file_integrity():
    assert GOLDEN["format"] == META["format"] == common.GOLDEN_FORMAT
    assert common.file_sha256(common.GOLDEN_PT) == META["pt_sha256"]
    assert [c["name"] for c in META["cases"]] == list(CASES)
    assert META["num_cases"] == len(CASES)


def test_case_sha256():
    recorded = {c["name"]: c["sha256"] for c in META["cases"]}
    for name, case in CASES.items():
        assert case["sha256"] == recorded[name], name
        assert common.case_sha256(case) == recorded[name], (
            f"{name}: fixture case content corrupted"
        )


def test_inventory_covers_every_kernel_and_e2e():
    kernels = {c["kernel"] for c in CASES.values()}
    assert kernels == set(common.KERNEL_NAMES) | {"rejection_sample"}
    for k in common.KERNEL_NAMES:
        assert any(
            c["kernel"] == k and c["source"].startswith("e2e_dump:")
            for c in CASES.values()
        ), f"no real-dump case for {k}"


def _check(name, launch):
    case = CASES[name]
    args = common.clone_args(case["args"])
    launch(case["kernel"], case["grid"], args, dict(case["kwargs"]))
    out = args[0]
    exp = case["output"]
    assert out.dtype == exp.dtype and out.shape == exp.shape
    if not torch.equal(out, exp):
        bad = (out != exp).nonzero()[:8].tolist()
        pytest.fail(
            f"{name}: Numba != Triton ({case['ref']}) at {bad}: "
            f"got {out.flatten()[:16].tolist()} exp {exp.flatten()[:16].tolist()}"
        )
    for i, (a, b) in enumerate(zip(args[1:], case["args"][1:], strict=False), start=1):
        if isinstance(b, torch.Tensor):
            assert torch.equal(a, b), f"{name}: input arg{i} mutated"


@pytest.mark.parametrize("name", KERNEL_CASES)
def test_golden_prod_wrapper(name, wrappers):
    _check(name, lambda k, g, a, kw: common.launch_wrapper(wrappers, k, g, a, kw))


@pytest.mark.parametrize("name", KERNEL_CASES)
def test_golden_raw_kernel(name):
    _check(name, common.launch_raw)


@pytest.mark.parametrize("name", E2E_CASES)
def test_golden_rejection_sample(name, numba_installed):
    from vllm.v1.sample import rejection_sampler as rs

    from vllm_qaic.v1.sample import rejection_sampler_numba as rsn

    assert all(getattr(rs, n) is not rsn._originals[n] for n in common.KERNEL_NAMES)
    case = CASES[name]
    before = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in case["inputs"].items()
    }
    out = common.run_rejection_sample(case["inputs"])
    assert torch.equal(out, case["output"]), (
        f"{name}: Numba rejection_sample != Triton ({case['ref']}):\n"
        f"{out.tolist()}\nvs\n{case['output'].tolist()}"
    )
    for k, v in before.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(case["inputs"][k], v), f"{name}: input {k} mutated"

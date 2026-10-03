# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""CPU-only tests for the QAIC AOT rejection-sampler backend selector.

Bit-equivalence of the Numba kernels against triton-cpu is covered by
experiments/test_numba_rejection_kernels{,_prod}.py; these tests cover the
selector, install/uninstall, validation guards and debug instrumentation.
"""

import json

import pytest
import torch

pytest.importorskip("numba")
import vllm.v1.sample.rejection_sampler as rs  # noqa: E402

from vllm_qaic.v1.sample import rejection_sampler_numba as rsn  # noqa: E402

UPSTREAM = {n: getattr(rs, n) for n in rsn.KERNEL_NAMES}


@pytest.fixture(autouse=True)
def _restore(monkeypatch, tmp_path):
    for var in (
        rsn.IMPL_ENV,
        "VLLM_QAIC_RS_COUNTERS",
        "VLLM_QAIC_RS_COUNTERS_DIR",
        "VLLM_QAIC_RS_DUMP",
        "VLLM_QAIC_RS_DUMP_STEPS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba_cache"))
    rsn.uninstall()
    # Captured per test: other modules (vllm_qaic.patch) may legitimately
    # replace RejectionSampler.forward at import time.
    forward = rs.RejectionSampler.forward
    yield forward
    rsn.uninstall()
    for n, k in UPSTREAM.items():
        assert getattr(rs, n) is k
    assert rs.RejectionSampler.forward is forward


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "triton"),
        ("", "triton"),
        ("triton", "triton"),
        ("NUMBA", "numba"),
        (" numba ", "numba"),
    ],
)
def test_selector(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(rsn.IMPL_ENV, value)
    assert rsn.selected_implementation() == expected


@pytest.mark.parametrize("value", ["pytorch", "hybrid", "triton-cpu"])
def test_selector_invalid_raises(monkeypatch, value):
    monkeypatch.setenv(rsn.IMPL_ENV, value)
    with pytest.raises(RuntimeError, match=rsn.IMPL_ENV):
        rsn.selected_implementation()
    with pytest.raises(RuntimeError):
        rsn.install()


def test_envs_declares_selector(monkeypatch):
    import vllm_qaic.envs as envs

    assert envs.VLLM_QAIC_AOT_REJECTION_SAMPLER_IMPL == "triton"
    monkeypatch.setenv(rsn.IMPL_ENV, "Numba")
    assert envs.VLLM_QAIC_AOT_REJECTION_SAMPLER_IMPL == "numba"


def test_triton_default_leaves_upstream_untouched(_restore):
    assert rsn.install() == "triton"
    for n, k in UPSTREAM.items():
        assert getattr(rs, n) is k
    assert rs.RejectionSampler.forward is _restore


def test_numba_install_swaps_and_uninstall_restores(monkeypatch):
    monkeypatch.setenv(rsn.IMPL_ENV, "numba")
    assert rsn.install() == "numba"
    for n in rsn.KERNEL_NAMES:
        assert getattr(rs, n) is not UPSTREAM[n]
        assert isinstance(getattr(rs, n), rsn._NumbaKernel)
    rsn.uninstall()
    for n, k in UPSTREAM.items():
        assert getattr(rs, n) is k


def test_install_idempotent_and_conflict_raises(monkeypatch):
    monkeypatch.setenv(rsn.IMPL_ENV, "numba")
    rsn.install()
    first = rs.expand_kernel
    rsn.install()
    assert rs.expand_kernel is first
    with pytest.raises(RuntimeError, match="already installed"):
        rsn.install("triton")


def test_expand_matches_reference_through_upstream_helper(monkeypatch):
    monkeypatch.setenv(rsn.IMPL_ENV, "numba")
    rsn.install()
    temp = torch.tensor([0.0, 0.7, 1.3], dtype=torch.float32)
    cu = torch.tensor([2, 2, 5], dtype=torch.int32)
    out = rs.expand_batch_to_tokens(temp, cu, 5, replace_from=0, replace_to=1)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([1.0, 1.0, 1.3, 1.3, 1.3]))
    top_k = torch.tensor([5, 50, 7], dtype=torch.int32)
    out = rs.expand_batch_to_tokens(top_k, cu, 5)
    assert out.dtype == torch.int32
    assert out.tolist() == [5, 5, 7, 7, 7]


def _greedy_args(draft_dtype=torch.int64):
    cu = torch.tensor([2, 4], dtype=torch.int32)
    draft = torch.tensor([3, 4, 5, 6], dtype=draft_dtype)
    argmax = torch.tensor([3, 9, 5, 6], dtype=torch.int64)
    bonus = torch.tensor([[11], [12]], dtype=torch.int32)
    out = torch.full((2, 3), -1, dtype=torch.int32)
    return [out, cu, draft, argmax, bonus, None, 2, None, None]


def test_greedy_int64_production_dtypes(monkeypatch):
    monkeypatch.setenv(rsn.IMPL_ENV, "numba")
    rsn.install()
    args = _greedy_args()
    rs.rejection_greedy_sample_kernel[(2,)](*args, SYNTHETIC_MODE=False)
    assert args[0].tolist() == [[3, 9, -1], [5, 6, 12]]


@pytest.mark.parametrize(
    "mutate,exc",
    [
        (lambda a: a.__setitem__(2, a[2].to(torch.float32)), TypeError),
        (lambda a: a.__setitem__(3, a[3].to(torch.int16)), TypeError),
        (
            lambda a: a.__setitem__(0, torch.full((3, 2), -1, dtype=torch.int32).t()),
            ValueError,
        ),
        (
            lambda a: a.__setitem__(
                2, torch.empty(4, dtype=torch.int64, device="meta")
            ),
            ValueError,
        ),
    ],
)
def test_guards_raise(monkeypatch, mutate, exc):
    monkeypatch.setenv(rsn.IMPL_ENV, "numba")
    rsn.install()
    args = _greedy_args()
    mutate(args)
    with pytest.raises(exc):
        rs.rejection_greedy_sample_kernel[(2,)](*args, SYNTHETIC_MODE=False)


def test_recovered_rejects_wrong_inv_q_dtype(monkeypatch):
    monkeypatch.setenv(rsn.IMPL_ENV, "numba")
    rsn.install()
    cu = torch.tensor([2], dtype=torch.int32)
    draft = torch.zeros(2, dtype=torch.int64)
    probs = torch.softmax(torch.randn(2, 16), -1)
    inv_q = torch.ones(1, 16, dtype=torch.float32)
    with pytest.raises(TypeError, match="inv_q"):
        rs.sample_recovered_tokens_kernel[(1, 2)](
            torch.empty_like(draft),
            cu,
            draft,
            None,
            probs,
            inv_q,
            16,
            8192,
            NO_DRAFT_PROBS=True,
            USE_FP64_GUMBEL=True,
        )


def test_prewarm_noop_for_triton():
    rsn.install("triton")
    assert rsn.prewarm() == 0.0


def test_prewarm_numba(monkeypatch):
    monkeypatch.setenv(rsn.IMPL_ENV, "numba")
    rsn.install()
    assert rsn.prewarm() > 0.0


@pytest.mark.parametrize("impl", ["triton", "numba"])
def test_counters_and_dump_round_trip(_restore, monkeypatch, tmp_path, impl):
    monkeypatch.setenv(rsn.IMPL_ENV, impl)
    monkeypatch.setenv("VLLM_QAIC_RS_COUNTERS", "1")
    monkeypatch.setenv("VLLM_QAIC_RS_COUNTERS_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("VLLM_QAIC_RS_DUMP", str(tmp_path / "d"))
    monkeypatch.setenv("VLLM_QAIC_RS_DUMP_STEPS", "1")
    rsn.install()
    assert rs.RejectionSampler.forward is not _restore
    for _ in range(3):
        args = _greedy_args()
        rs.rejection_greedy_sample_kernel[(2,)](*args, SYNTHETIC_MODE=False)
    rsn._instrumentation.flush()
    (counter_file,) = (tmp_path / "c").glob("rs_counters_*.json")
    payload = json.loads(counter_file.read_text())
    assert payload["impl"] == impl
    (key,) = payload["kernels"]
    assert key.startswith("rejection_greedy_sample|SYNTHETIC_MODE=0|mask=none")
    assert "int64" in key and payload["kernels"][key]["calls"] == 3
    (dump_file,) = (tmp_path / "d").glob("*.pt")
    dump = torch.load(dump_file, weights_only=False)
    assert dump["kernel"] == "rejection_greedy_sample_kernel"
    assert dump["impl"] == impl and dump["grid"] == (2,)
    assert dump["args"][0].tolist() == [[-1, -1, -1]] * 2  # cloned pre-call
    assert dump["out_after"].tolist() == [[3, 9, -1], [5, 6, 12]]


def test_prewarm_does_not_consume_global_rng(monkeypatch):
    # The worker seeds the global RNG before warm-up; prewarm must not shift
    # the stream unseeded requests later sample from.
    monkeypatch.setenv(rsn.IMPL_ENV, "numba")
    rsn.install()
    torch.manual_seed(1234)
    expected = torch.rand(8)
    torch.manual_seed(1234)
    rsn.prewarm()
    assert torch.equal(torch.rand(8), expected)

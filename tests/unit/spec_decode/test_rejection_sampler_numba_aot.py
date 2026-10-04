# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""CPU-only tests for the QAIC AOT Numba rejection sampler.

Bit-equivalence of the Numba kernels against triton-cpu is covered by
tests/unit/spec_decode/rejection_parity/; these tests cover
install/uninstall, validation guards and debug instrumentation.
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
        rsn._REMOVED_IMPL_ENV,
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


def test_envs_no_longer_declares_selector():
    import vllm_qaic.envs as envs

    assert rsn._REMOVED_IMPL_ENV not in envs.qaic_environment_variables


def test_install_swaps_and_uninstall_restores(_restore):
    assert rsn.install() == "numba"
    for n in rsn.KERNEL_NAMES:
        assert getattr(rs, n) is not UPSTREAM[n]
        assert isinstance(getattr(rs, n), rsn._NumbaKernel)
    assert rs.RejectionSampler.forward is _restore
    rsn.uninstall()
    for n, k in UPSTREAM.items():
        assert getattr(rs, n) is k


def test_install_idempotent():
    rsn.install()
    first = rs.expand_kernel
    rsn.install()
    assert rs.expand_kernel is first


@pytest.mark.parametrize("value", ["triton", "numba"])
def test_removed_selector_warns_and_is_ignored(monkeypatch, value):
    monkeypatch.setenv(rsn._REMOVED_IMPL_ENV, value)
    warnings = []
    monkeypatch.setattr(rsn.logger, "warning", lambda *a, **k: warnings.append(a))
    assert rsn.install() == "numba"
    assert isinstance(rs.expand_kernel, rsn._NumbaKernel)
    assert len(warnings) == 1 and rsn._REMOVED_IMPL_ENV in warnings[0]


def test_install_fails_loudly_without_numba(monkeypatch):
    def _broken(threads):
        raise ImportError("No module named 'numba'")

    monkeypatch.setattr(rsn, "_configure_numba", _broken)
    with pytest.raises(RuntimeError, match="requires Numba"):
        rsn.install()
    # Nothing was swapped, so the fixture's restore check still holds.
    assert not rsn._installed


def test_expand_matches_reference_through_upstream_helper(monkeypatch):
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
    rsn.install()
    args = _greedy_args()
    mutate(args)
    with pytest.raises(exc):
        rs.rejection_greedy_sample_kernel[(2,)](*args, SYNTHETIC_MODE=False)


def test_recovered_rejects_wrong_inv_q_dtype(monkeypatch):
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


def test_prewarm_noop_before_install():
    assert rsn.prewarm() == 0.0


def test_prewarm_numba(monkeypatch):
    rsn.install()
    assert rsn.prewarm() > 0.0


def test_counters_and_dump_round_trip(_restore, monkeypatch, tmp_path):
    impl = "numba"
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
    rsn.install()
    torch.manual_seed(1234)
    expected = torch.rand(8)
    torch.manual_seed(1234)
    rsn.prewarm()
    assert torch.equal(torch.rand(8), expected)

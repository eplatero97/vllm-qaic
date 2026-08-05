# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Unit tests for the PyTorch rejection-sampler shim.

These tests target the pure-PyTorch kernel replacements in
``vllm_qaic.v1.sample.rejection_sampler_shim`` that enable ngram/suffix
speculative decoding in PYT (eager) mode on QAIC.

Every kernel test is parametrized over ``device`` (``cpu`` and, when a QAIC
device is available, ``qaic``).  CPU is the reference; running on ``qaic``
proves the ops actually dispatch and compute on the NPU — which is the whole
reason the shim exists.  In particular the recovered-tokens test exercises the
``scatter_`` workaround that replaced 2-D advanced indexing (which segfaulted
on QAIC).

Run::

    .venv_eager/bin/python -m pytest -s \
        tests/test_qaic/spec_decode/test_rejection_sampler_shim.py -v
"""

import pytest
import torch

from vllm_qaic.v1.sample.rejection_sampler_shim import (
    _GridLaunchable,
    _expand_kernel_pyt,
    _rejection_greedy_sample_kernel_pyt,
    _rejection_random_sample_kernel_pyt,
    _sample_recovered_tokens_kernel_pyt,
    generate_uniform_probs,
    install,
)

# vLLM fills unwritten output slots with this sentinel.
PLACEHOLDER_TOKEN_ID = -1

# Discover available devices.  torch.qaic only exists once torch_qaic is
# imported; fall back to cpu-only when it is absent (e.g. AOT env).
_DEVICES = ["cpu"]
try:
    import torch_qaic  # noqa: F401

    if torch.qaic.device_count() > 0:
        _DEVICES.append("qaic")
except Exception:
    pass


@pytest.fixture(params=_DEVICES)
def device(request) -> torch.device:
    return torch.device(request.param)


# ---------------------------------------------------------------------------
# _GridLaunchable (device-agnostic)
# ---------------------------------------------------------------------------


def test_grid_launchable_discards_grid_and_forwards_args():
    seen = []

    def fn(a, b, KW=None):
        seen.append((a, b, KW))
        return "ret"

    wrapped = _GridLaunchable(fn)

    # kernel[(grid,)](*args, KW=v) — grid must be discarded.
    assert wrapped[(4,)](1, 2, KW=3) == "ret"
    # Direct call also works.
    assert wrapped(10, 20, KW=30) == "ret"

    assert seen == [(1, 2, 3), (10, 20, 30)]
    # __name__ is propagated for readable logging.
    assert wrapped.__name__ == "fn"


# ---------------------------------------------------------------------------
# expand_kernel
# ---------------------------------------------------------------------------


def test_expand_kernel_basic(device):
    x = torch.tensor([10, 20, 30], device=device)
    cu = torch.tensor([2, 5, 6], device=device)
    out = torch.zeros(6, dtype=torch.long, device=device)

    _expand_kernel_pyt(out, x, cu, replace_from=0, replace_to=0)

    assert out.cpu().tolist() == [10, 10, 20, 20, 20, 30]


def test_expand_kernel_replace(device):
    x = torch.tensor([0, 5, 7], device=device)
    cu = torch.tensor([2, 5, 6], device=device)
    out = torch.zeros(6, dtype=torch.long, device=device)

    _expand_kernel_pyt(out, x, cu, replace_from=0, replace_to=99)

    assert out.cpu().tolist() == [99, 99, 5, 5, 5, 7]


def test_expand_kernel_empty_batch(device):
    x = torch.zeros(0, dtype=torch.long, device=device)
    cu = torch.zeros(0, dtype=torch.long, device=device)
    out = torch.zeros(0, dtype=torch.long, device=device)

    # Should be a no-op without error.
    _expand_kernel_pyt(out, x, cu, replace_from=0, replace_to=0)

    assert out.cpu().tolist() == []


# ---------------------------------------------------------------------------
# rejection_greedy_sample_kernel  (SYNTHETIC_MODE=False, is_greedy=None)
# ---------------------------------------------------------------------------


def _greedy_output(batch_size, max_spec_len, device):
    return torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )


def test_greedy_all_accept_appends_bonus(device):
    # Single request, 2 draft tokens, both match target argmax.
    out = _greedy_output(1, 3, device)
    cu = torch.tensor([2], device=device)
    drafts = torch.tensor([5, 5], device=device)
    target_argmax = torch.tensor([5, 5], device=device)
    bonus = torch.tensor([99], device=device)

    _rejection_greedy_sample_kernel_pyt(
        out,
        cu,
        drafts,
        target_argmax,
        bonus,
        None,
        3,
        None,
        None,
        SYNTHETIC_MODE=False,
    )

    # Both accepted → target tokens, then bonus at position num_draft (2).
    assert out.cpu().tolist()[0] == [5, 5, 99, PLACEHOLDER_TOKEN_ID]


def test_greedy_first_mismatch_stops_no_bonus(device):
    # Single request, first draft token mismatches → only target[0] stored.
    out = _greedy_output(1, 3, device)
    cu = torch.tensor([2], device=device)
    drafts = torch.tensor([7, 8], device=device)
    target_argmax = torch.tensor([9, 8], device=device)  # pos0 mismatches
    bonus = torch.tensor([99], device=device)

    _rejection_greedy_sample_kernel_pyt(
        out,
        cu,
        drafts,
        target_argmax,
        bonus,
        None,
        3,
        None,
        None,
        SYNTHETIC_MODE=False,
    )

    # pos0 = target argmax (9), then rejected → no further writes, no bonus.
    assert out.cpu().tolist()[0] == [
        9,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
    ]


def test_greedy_multi_request_batch(device):
    # req0: all match (bonus appended); req1: mismatch at pos0.
    out = _greedy_output(2, 3, device)
    cu = torch.tensor([2, 4], device=device)
    drafts = torch.tensor([5, 5, 7, 8], device=device)
    target_argmax = torch.tensor([5, 5, 9, 8], device=device)
    bonus = torch.tensor([99, 88], device=device)

    _rejection_greedy_sample_kernel_pyt(
        out,
        cu,
        drafts,
        target_argmax,
        bonus,
        None,
        3,
        None,
        None,
        SYNTHETIC_MODE=False,
    )

    rows = out.cpu().tolist()
    assert rows[0] == [5, 5, 99, PLACEHOLDER_TOKEN_ID]
    assert rows[1] == [
        9,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
    ]


# ---------------------------------------------------------------------------
# rejection_random_sample_kernel  (NO_DRAFT_PROBS=True)
# ---------------------------------------------------------------------------


def _random_probs(num_tokens, vocab, device, fill=0.5):
    return torch.full((num_tokens, vocab), fill, dtype=torch.float32, device=device)


def test_random_uniform_zero_accepts_all(device):
    # uniform=0 → accepted = target_prob/1.0 >= 0 → always true.
    out = _greedy_output(1, 3, device)
    cu = torch.tensor([2], device=device)
    drafts = torch.tensor([1, 2], device=device)
    vocab = 4
    target_probs = _random_probs(2, vocab, device, fill=0.5)
    bonus = torch.tensor([99], device=device)
    recovered = torch.tensor([0, 0], device=device)
    uniform = torch.zeros(2, dtype=torch.float32, device=device)
    is_greedy = torch.tensor([False], device=device)

    _rejection_random_sample_kernel_pyt(
        out,
        cu,
        drafts,
        None,
        target_probs,
        bonus,
        recovered,
        uniform,
        is_greedy,
        3,
        vocab,
        None,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
    )

    # All draft tokens accepted, bonus appended.
    assert out.cpu().tolist()[0] == [1, 2, 99, PLACEHOLDER_TOKEN_ID]


def test_random_uniform_one_rejects_and_recovers(device):
    # uniform=1, target_prob=0.5 < 1 → reject at pos0 → recovered token.
    out = _greedy_output(1, 3, device)
    cu = torch.tensor([2], device=device)
    drafts = torch.tensor([1, 2], device=device)
    vocab = 4
    target_probs = _random_probs(2, vocab, device, fill=0.5)
    bonus = torch.tensor([99], device=device)
    recovered = torch.tensor([42, 43], device=device)
    uniform = torch.ones(2, dtype=torch.float32, device=device)
    is_greedy = torch.tensor([False], device=device)

    _rejection_random_sample_kernel_pyt(
        out,
        cu,
        drafts,
        None,
        target_probs,
        bonus,
        recovered,
        uniform,
        is_greedy,
        3,
        vocab,
        None,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
    )

    # pos0 rejected → recovered[0]=42, then stop. No bonus.
    assert out.cpu().tolist()[0] == [
        42,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
    ]


def test_random_skips_greedy_requests(device):
    # is_greedy=True → row must be left untouched (handled by greedy kernel).
    out = _greedy_output(1, 3, device)
    cu = torch.tensor([2], device=device)
    drafts = torch.tensor([1, 2], device=device)
    vocab = 4
    target_probs = _random_probs(2, vocab, device, fill=0.5)
    bonus = torch.tensor([99], device=device)
    recovered = torch.tensor([0, 0], device=device)
    uniform = torch.zeros(2, dtype=torch.float32, device=device)
    is_greedy = torch.tensor([True], device=device)

    _rejection_random_sample_kernel_pyt(
        out,
        cu,
        drafts,
        None,
        target_probs,
        bonus,
        recovered,
        uniform,
        is_greedy,
        3,
        vocab,
        None,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
    )

    assert out.cpu().tolist()[0] == [PLACEHOLDER_TOKEN_ID] * 4


# ---------------------------------------------------------------------------
# sample_recovered_tokens_kernel
# ---------------------------------------------------------------------------


def test_recovered_no_draft_probs_zeros_draft_column(device):
    # NO_DRAFT_PROBS=True: draft-token column is zeroed, argmax over the rest.
    vocab = 5
    cu = torch.tensor([1, 2], device=device)
    draft_toks = torch.tensor([2, 0], device=device)
    target_probs = torch.tensor(
        [
            [0.1, 0.1, 0.9, 0.6, 0.2],  # req0 draft=2: zero col2 → argmax=3
            [0.9, 0.2, 0.3, 0.1, 0.4],  # req1 draft=0: zero col0 → argmax=4
        ],
        dtype=torch.float32,
        device=device,
    )
    inv_q = torch.ones(2, vocab, dtype=torch.float32, device=device)
    out = torch.zeros(2, dtype=torch.int32, device=device)

    _sample_recovered_tokens_kernel_pyt(
        out,
        cu,
        draft_toks,
        None,
        target_probs,
        inv_q,
        vocab,
        NO_DRAFT_PROBS=True,
        USE_FP64_GUMBEL=False,
    )

    assert out.cpu().tolist() == [3, 4]


def test_recovered_with_draft_probs_uses_clamped_diff(device):
    # NO_DRAFT_PROBS=False: prob = max(target - draft, 0), then argmax.
    vocab = 4
    cu = torch.tensor([1], device=device)
    draft_toks = torch.tensor([0], device=device)
    target_probs = torch.tensor(
        [[0.1, 0.2, 0.7, 0.0]], dtype=torch.float32, device=device
    )
    draft_probs = torch.tensor(
        [[0.0, 0.5, 0.1, 0.0]], dtype=torch.float32, device=device
    )
    # diff = [0.1, -0.3→0, 0.6, 0.0] → argmax = 2
    inv_q = torch.ones(1, vocab, dtype=torch.float32, device=device)
    out = torch.zeros(1, dtype=torch.int32, device=device)

    _sample_recovered_tokens_kernel_pyt(
        out,
        cu,
        draft_toks,
        draft_probs,
        target_probs,
        inv_q,
        vocab,
        NO_DRAFT_PROBS=False,
        USE_FP64_GUMBEL=False,
    )

    assert out.cpu().tolist() == [2]


def test_recovered_skips_zero_draft_request(device):
    # A request with num_draft==0 must be skipped (output slot untouched).
    vocab = 4
    cu = torch.tensor([0, 1], device=device)  # req0 has 0 draft tokens
    draft_toks = torch.tensor([3], device=device)  # only req1's token
    target_probs = torch.tensor(
        [[0.5, 0.2, 0.9, 0.1]], dtype=torch.float32, device=device
    )
    inv_q = torch.ones(2, vocab, dtype=torch.float32, device=device)
    out = torch.full((1,), -7, dtype=torch.int32, device=device)

    _sample_recovered_tokens_kernel_pyt(
        out,
        cu,
        draft_toks,
        None,
        target_probs,
        inv_q,
        vocab,
        NO_DRAFT_PROBS=True,
        USE_FP64_GUMBEL=False,
    )

    # req1 draft=3 → zero col3 → argmax over [0.5,0.2,0.9,0] = 2.
    assert out.cpu().tolist() == [2]


# ---------------------------------------------------------------------------
# generate_uniform_probs
# ---------------------------------------------------------------------------


def test_generate_uniform_probs_shape_dtype_range(device):
    u = generate_uniform_probs(6, [2, 4], {}, device)

    assert u.shape == (6,)
    # QAIC does not support float64 — the shim must use float32.
    assert u.dtype == torch.float32
    u_cpu = u.cpu()
    assert (u_cpu >= 0).all() and (u_cpu <= 1).all()


def test_generate_uniform_probs_seeded_is_deterministic():
    # Generator seeding is CPU-only; verify reproducibility on cpu.
    dev = torch.device("cpu")
    g1 = torch.Generator(device="cpu").manual_seed(1234)
    g2 = torch.Generator(device="cpu").manual_seed(1234)

    u1 = generate_uniform_probs(4, [4], {0: g1}, dev)
    u2 = generate_uniform_probs(4, [4], {0: g2}, dev)

    assert torch.equal(u1, u2)


# ---------------------------------------------------------------------------
# install()  (device-agnostic)
# ---------------------------------------------------------------------------


def test_install_patches_module_and_is_idempotent():
    import vllm_qaic.v1.sample.rejection_sampler_shim as shim
    import vllm.v1.sample.rejection_sampler as rs

    install()

    assert rs.expand_kernel is shim.expand_kernel
    assert rs.rejection_greedy_sample_kernel is shim.rejection_greedy_sample_kernel
    assert rs.rejection_random_sample_kernel is shim.rejection_random_sample_kernel
    assert rs.sample_recovered_tokens_kernel is shim.sample_recovered_tokens_kernel
    assert rs.generate_uniform_probs is shim.generate_uniform_probs

    # Second call is a no-op (guarded by _shim_installed).
    install()
    assert rs.expand_kernel is shim.expand_kernel

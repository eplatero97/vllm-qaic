# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Benchmark-harness oracle cases: Triton and Numba vs an independent scalar oracle.

Ports the 93-case correctness check of the original (uncommitted) AOT
rejection-sampler benchmark harness.  Those cases are NOT a subset of
``test_triton_parity_kernels.py`` / ``test_triton_parity_prod.py``: the
fixtures are near-constant (linspace inv_q, 0.75/0.001 probs), uniform probs
are fp32, draft ids are int32, behaviours include first/late rejection, and
correctness is judged against a pure-Python scalar oracle rather than Triton
alone.  Fixture builders, oracles and launchers below are copied verbatim
(``Case``/``make_fixture``/``scalar_*``/``_launch_*``/``_cases``); the timing
loop and the PyTorch backend are dropped and the production wrapper
(``install()``) is added as a backend.

Tier A (``triton_parity``): skipped without triton-cpu.

    TRITON_CPU_BACKEND=1 .venv_aot/bin/python -m pytest -s -q \
        tests/test_qaic/spec_decode/rejection_parity/test_triton_parity_harness.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import _parity_common as common  # noqa: E402  (captures upstream kernels first)
import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_qaic.v1.sample import numba_rejection_kernels as nrk  # noqa: E402

pytestmark = pytest.mark.triton_parity

_KERNEL_GLOBAL = {
    "expand": "expand_kernel",
    "greedy": "rejection_greedy_sample_kernel",
    "random": "rejection_random_sample_kernel",
    "recovered": "sample_recovered_tokens_kernel",
}


class _Kernels:
    """``_launch_triton`` attribute layout, bound to an arbitrary backend."""

    def __init__(self, table: dict[str, Any]):
        for short, name in _KERNEL_GLOBAL.items():
            setattr(self, short, table[name])


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    max_spec_len: int
    vocab_size: int
    count_pattern: str = "ragged"
    behavior: str = "accepted"
    no_draft_probs: bool = True
    synthetic: bool = False
    use_fp64: bool = False


@dataclass
class Fixture:
    case: Case
    counts: torch.Tensor
    draft_ids: torch.Tensor
    target_argmax: torch.Tensor
    bonus_ids: torch.Tensor
    is_greedy: torch.Tensor
    uniform: torch.Tensor
    rates: torch.Tensor
    draft_probs: torch.Tensor | None
    target_probs: torch.Tensor
    recovered_ids: torch.Tensor
    inv_q: torch.Tensor
    replace_from: int
    replace_to: int

    @property
    def total_tokens(self) -> int:
        return int(self.counts[-1].item()) if self.counts.numel() else 0


def _cumulative_counts(counts: list[int]) -> torch.Tensor:
    total = 0
    cumulative = []
    for count in counts:
        total += count
        cumulative.append(total)
    return torch.tensor(cumulative, dtype=torch.int32)


def _counts(case: Case) -> list[int]:
    if case.count_pattern == "full":
        return [case.max_spec_len] * case.batch
    if case.count_pattern == "zero_first":
        return [0] + [case.max_spec_len] * (case.batch - 1)
    if case.count_pattern == "ragged":
        return [
            0
            if index > 0 and index % 7 == 0
            else max(1, case.max_spec_len - (index % 3))
            for index in range(case.batch)
        ]
    raise ValueError(f"unknown count pattern: {case.count_pattern}")


def _first_active(counts: list[int]) -> int | None:
    offset = 0
    for count in counts:
        if count:
            return offset
        offset += count
    return None


def _last_active(counts: list[int]) -> int | None:
    offset = 0
    last = None
    for count in counts:
        if count:
            last = offset + count - 1
        offset += count
    return last


def make_fixture(case: Case) -> Fixture:
    counts_list = _counts(case)
    counts = _cumulative_counts(counts_list)
    total = int(counts[-1].item()) if counts.numel() else 0
    vocab = case.vocab_size
    draft_ids = (torch.arange(total, dtype=torch.int32) * 17 + 3) % vocab
    target_argmax = draft_ids.clone()
    bonus_ids = (torch.arange(case.batch, dtype=torch.int32) * 19 + 11) % vocab
    is_greedy = torch.ones(case.batch, dtype=torch.bool)
    uniform = torch.full((total,), 0.1, dtype=torch.float32)
    rates = torch.ones(case.max_spec_len, dtype=torch.float32)

    if case.behavior in {"first_reject", "late_reject"}:
        position = (
            _first_active(counts_list)
            if case.behavior == "first_reject"
            else _last_active(counts_list)
        )
        assert position is not None
        target_argmax[position] = (draft_ids[position] + 1) % vocab
    if case.behavior.startswith("random_"):
        is_greedy.fill_(False)
    if case.behavior == "mixed_requests":
        is_greedy[1::2] = False
    if case.behavior == "non_greedy":
        is_greedy.fill_(False)
    if "synthetic" in case.behavior:
        rates.fill_(1.0)
        rates[0] = 0.0
        uniform.fill_(0.1)
    if case.behavior == "random_first_reject":
        position = _first_active(counts_list)
        assert position is not None
        uniform[position] = 0.9
    if case.behavior == "random_late_reject":
        position = _last_active(counts_list)
        assert position is not None
        uniform[position] = 0.9
    if case.behavior == "random_mixed_requests":
        is_greedy[1::2] = True

    target_probs = torch.full((total, vocab), 0.001, dtype=torch.float32)
    target_probs[torch.arange(total), draft_ids.to(torch.int64)] = 0.75
    draft_probs: torch.Tensor | None = None
    if not case.no_draft_probs:
        draft_probs = torch.full((total, vocab), 0.001, dtype=torch.float32)
        draft_probs[torch.arange(total), draft_ids.to(torch.int64)] = 0.5

    # Make the recovered-token argmax deterministic while avoiding widespread
    # ties.  The same values are used by both candidate implementations.
    inv_q_dtype = torch.float64 if case.use_fp64 else torch.float32
    inv_q = torch.linspace(0.5, 1.5, vocab, dtype=inv_q_dtype).repeat(case.batch, 1)
    recovered_ids = torch.zeros(total, dtype=torch.int32)
    replace_from = 7
    replace_to = 70
    return Fixture(
        case=case,
        counts=counts,
        draft_ids=draft_ids,
        target_argmax=target_argmax,
        bonus_ids=bonus_ids,
        is_greedy=is_greedy,
        uniform=uniform,
        rates=rates,
        draft_probs=draft_probs,
        target_probs=target_probs,
        recovered_ids=recovered_ids,
        inv_q=inv_q,
        replace_from=replace_from,
        replace_to=replace_to,
    )


def scalar_expand(fixture: Fixture) -> list[int]:
    result: list[int] = []
    counts = fixture.counts.tolist()
    previous = 0
    for request, end in enumerate(counts):
        value = request
        if value == fixture.replace_from:
            value = fixture.replace_to
        result.extend([value] * (end - previous))
        previous = end
    return result


def scalar_rejection(fixture: Fixture, greedy: bool) -> list[list[int]]:
    case = fixture.case
    counts = fixture.counts.tolist()
    starts = [0] + counts[:-1]
    result = [[-1] * (case.max_spec_len + 1) for _ in range(case.batch)]
    for request, end in enumerate(counts):
        if (greedy and not bool(fixture.is_greedy[request])) or (
            not greedy and bool(fixture.is_greedy[request])
        ):
            continue
        rejected = False
        length = end - starts[request]
        for position in range(length):
            index = starts[request] + position
            draft = int(fixture.draft_ids[index])
            if greedy:
                if case.synthetic:
                    accepted = float(fixture.uniform[index]) < float(
                        fixture.rates[position]
                    )
                    token = draft if accepted else int(fixture.target_argmax[index])
                else:
                    accepted = draft == int(fixture.target_argmax[index])
                    token = int(fixture.target_argmax[index])
            else:
                if case.synthetic:
                    accepted = float(fixture.uniform[index]) < float(
                        fixture.rates[position]
                    )
                else:
                    if case.no_draft_probs:
                        draft_prob = 1.0
                    else:
                        assert fixture.draft_probs is not None
                        draft_prob = float(fixture.draft_probs[index, draft])
                    target_prob = float(fixture.target_probs[index, draft])
                    accepted = draft_prob > 0 and target_prob / draft_prob >= float(
                        fixture.uniform[index]
                    )
                token = draft if accepted else int(fixture.recovered_ids[index])
            if not rejected:
                result[request][position] = token
            rejected |= not accepted
        if not rejected:
            result[request][length] = int(fixture.bonus_ids[request])
    return result


def scalar_recovered(fixture: Fixture) -> list[int]:
    result = [-1] * fixture.total_tokens
    counts = fixture.counts.tolist()
    starts = [0] + counts[:-1]
    target_probs = fixture.target_probs.tolist()
    inv_q = fixture.inv_q.tolist()
    draft_probs = (
        fixture.draft_probs.tolist() if fixture.draft_probs is not None else None
    )
    draft_ids = fixture.draft_ids.tolist()
    for request, end in enumerate(counts):
        for index in range(starts[request], end):
            scores: list[float] = []
            draft = int(draft_ids[index])
            for vocab_id in range(fixture.case.vocab_size):
                if fixture.case.no_draft_probs:
                    probability = (
                        0.0 if vocab_id == draft else target_probs[index][vocab_id]
                    )
                else:
                    assert draft_probs is not None
                    probability = max(
                        target_probs[index][vocab_id] - draft_probs[index][vocab_id],
                        0.0,
                    )
                probability *= inv_q[request][vocab_id]
                scores.append(probability)
            result[index] = max(
                range(len(scores)), key=lambda vocab_id: scores[vocab_id]
            )
    return result


def _make_expected(fixture: Fixture, kernel: str) -> torch.Tensor:
    case = fixture.case  # noqa: F841 - verbatim copy of the benchmark oracle
    if kernel == "expand":
        expected = torch.tensor(scalar_expand(fixture), dtype=torch.int32)
    elif kernel == "greedy":
        rows = scalar_rejection(fixture, greedy=True)
        expected = torch.tensor(rows, dtype=torch.int32)
    elif kernel == "random":
        rows = scalar_rejection(fixture, greedy=False)
        expected = torch.tensor(rows, dtype=torch.int32)
    elif kernel == "recovered":
        expected = torch.tensor(scalar_recovered(fixture), dtype=torch.int32)
    else:
        raise ValueError(kernel)
    return expected


def _output(fixture: Fixture, kernel: str) -> torch.Tensor:
    case = fixture.case
    if kernel in {"greedy", "random"}:
        shape: tuple[int, ...] = (case.batch, case.max_spec_len + 1)
    else:
        shape = (fixture.total_tokens,)
    return torch.full(shape, -1, dtype=torch.int32)


def _launch_triton(
    fixture: Fixture, kernel: str, output: torch.Tensor, kernels: Any
) -> None:
    case = fixture.case
    if kernel == "expand":
        kernels.expand[(case.batch,)](
            output,
            torch.arange(case.batch, dtype=torch.int32),
            fixture.counts,
            fixture.replace_from,
            fixture.replace_to,
            MAX_NUM_TOKENS=case.max_spec_len,
        )
    elif kernel == "greedy":
        kernels.greedy[(case.batch,)](
            output,
            fixture.counts,
            fixture.draft_ids,
            fixture.target_argmax,
            fixture.bonus_ids,
            fixture.is_greedy,
            case.max_spec_len,
            fixture.uniform,
            fixture.rates,
            SYNTHETIC_MODE=case.synthetic,
        )
    elif kernel == "random":
        kernels.random[(case.batch,)](
            output,
            fixture.counts,
            fixture.draft_ids,
            fixture.draft_probs,
            fixture.target_probs,
            fixture.bonus_ids,
            fixture.recovered_ids,
            fixture.uniform,
            fixture.is_greedy,
            case.max_spec_len,
            case.vocab_size,
            fixture.rates,
            NO_DRAFT_PROBS=case.no_draft_probs,
            SYNTHETIC_MODE=case.synthetic,
        )
    elif kernel == "recovered":
        kernels.recovered[(case.batch, case.max_spec_len)](
            output,
            fixture.counts,
            fixture.draft_ids,
            fixture.draft_probs,
            fixture.target_probs,
            fixture.inv_q,
            case.vocab_size,
            BLOCK_SIZE=8192,
            NO_DRAFT_PROBS=case.no_draft_probs,
            USE_FP64_GUMBEL=case.use_fp64,
        )
    else:
        raise ValueError(kernel)


def _launch_numba(
    fixture: Fixture, kernel: str, output: torch.Tensor, parallel: bool
) -> None:
    case = fixture.case
    if kernel == "expand":
        # Mirrors the Triton launch, which also builds the input per call.
        nrk.expand_numba(
            output,
            torch.arange(case.batch, dtype=torch.int32),
            fixture.counts,
            fixture.replace_from,
            fixture.replace_to,
        )
    elif kernel == "greedy":
        nrk.greedy_numba(
            output,
            fixture.counts,
            fixture.draft_ids,
            fixture.target_argmax,
            fixture.bonus_ids,
            fixture.is_greedy,
            case.max_spec_len,
            fixture.uniform,
            fixture.rates,
            case.synthetic,
            parallel=parallel,
        )
    elif kernel == "random":
        nrk.random_numba(
            output,
            fixture.counts,
            fixture.draft_ids,
            fixture.draft_probs,
            fixture.target_probs,
            fixture.bonus_ids,
            fixture.recovered_ids,
            fixture.uniform,
            fixture.is_greedy,
            case.max_spec_len,
            case.vocab_size,
            fixture.rates,
            case.no_draft_probs,
            case.synthetic,
            parallel=parallel,
        )
    elif kernel == "recovered":
        nrk.recovered_numba(
            output,
            fixture.counts,
            fixture.draft_ids,
            fixture.draft_probs,
            fixture.target_probs,
            fixture.inv_q,
            case.vocab_size,
            case.no_draft_probs,
            parallel=parallel,
        )
    else:
        raise ValueError(kernel)


def _cases() -> dict[str, list[Case]]:
    scales = [
        ("small", 1, 1, 128),
        ("medium", 8, 4, 1024),
        ("large", 32, 8, 8192),
        ("wide_batch", 128, 16, 8192),
        ("wide_vocab", 8, 8, 32768),
    ]
    # Production-sized vocab (Llama-3 128256) for the only vocab-bound kernel.
    prod_scales = [("prod_vocab", 16, 4, 128256)]
    expand = [Case(name, batch, length, vocab) for name, batch, length, vocab in scales]
    greedy = [
        Case(f"{name}_{behavior}", batch, length, vocab, behavior=behavior)
        for name, batch, length, vocab in scales
        for behavior in ("accepted", "first_reject", "late_reject", "mixed_requests")
    ]
    greedy.extend(
        Case(
            f"{name}_synthetic_first_reject",
            batch,
            length,
            vocab,
            behavior="synthetic_first_reject",
            synthetic=True,
        )
        for name, batch, length, vocab in scales
    )
    random_cases = [
        Case(
            f"{name}_{behavior}_{'nodraft' if no_draft else 'draft'}",
            batch,
            length,
            vocab,
            behavior=behavior,
            no_draft_probs=no_draft,
        )
        for name, batch, length, vocab in scales
        for behavior in (
            "random_accepted",
            "random_first_reject",
            "random_late_reject",
            "random_mixed_requests",
        )
        for no_draft in (True, False)
    ]
    random_cases.extend(
        Case(
            f"{name}_random_synthetic_first_reject",
            batch,
            length,
            vocab,
            behavior="random_synthetic_first_reject",
            synthetic=True,
        )
        for name, batch, length, vocab in scales
    )
    recovered = [
        Case(
            f"{name}_{mode}_{'fp64' if fp64 else 'fp32'}",
            batch,
            length,
            vocab,
            no_draft_probs=no_draft,
            use_fp64=fp64,
        )
        for name, batch, length, vocab in scales
        for mode, no_draft in (("nodraft", True), ("draft", False))
        for fp64 in (
            (False, True) if name in {"small", "medium", "wide_vocab"} else (False,)
        )
    ]
    recovered.extend(
        Case(f"{name}_{mode}_fp32", batch, length, vocab, no_draft_probs=no_draft)
        for name, batch, length, vocab in prod_scales
        for mode, no_draft in (("nodraft", True), ("draft", False))
    )
    return {
        "expand": expand,
        "greedy": greedy,
        "random": random_cases,
        "recovered": recovered,
    }


_ALL_CASES = [(kernel, case) for kernel, cases in _cases().items() for case in cases]
assert len(_ALL_CASES) == 93, len(_ALL_CASES)
BACKENDS = ("triton", "numba", "numba_par", "prod")


@pytest.fixture(scope="module")
def wrappers():
    common.assert_triton_originals()
    w = common.production_wrappers()
    assert w is not None
    return w


def test_negative_control():
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "kernel,case", _ALL_CASES, ids=[f"{k}-{c.name}" for k, c in _ALL_CASES]
)
def test_harness_case(kernel, case, wrappers):
    fixture = make_fixture(case)
    expected = _make_expected(fixture, kernel)
    triton = _Kernels(common.ORIGINAL_KERNELS)
    prod = _Kernels(wrappers)
    for backend in BACKENDS:
        out = _output(fixture, kernel)
        if backend == "triton":
            _launch_triton(fixture, kernel, out, triton)
        elif backend == "prod":
            _launch_triton(fixture, kernel, out, prod)
        else:
            _launch_numba(fixture, kernel, out, parallel=backend == "numba_par")
        assert torch.equal(out, expected), (backend, kernel, case.name, out, expected)

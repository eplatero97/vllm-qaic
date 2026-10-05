# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Generate the tier-B golden fixtures for the Numba rejection sampler.

Writes ``tests/unit/spec_decode/rejection_parity/fixtures/rs_golden_v1.pt``
(deflate-compressed ``torch.save``; ``torch.load(weights_only=True)``-safe)
plus ``rs_golden_v1.json`` metadata.  Every stored output is produced by the
REAL upstream triton-cpu kernels, so this script refuses to run unless the
``vllm.v1.sample.rejection_sampler`` kernels are ``triton.runtime.jit.JITFunction``
objects (triton-cpu installed, ``TRITON_CPU_BACKEND=1``).

Cases (all deterministic):

1. ``dump:*`` -- curated real E2E launches from ``--dumps`` (default
   ``runs/identity_notriton/dumps``; written by ``VLLM_QAIC_RS_DUMP``): the
   smallest file per (kernel, constexpr kwargs, arg dtype signature, greedy
   mask none/tensor).  To fit the size budget, elements the kernel provably
   never reads are zeroed (random kernel: every ``target/draft_probs`` entry
   except ``[t, draft[t]]``; recovered kernel: ``inv_q`` rows of requests with
   no draft tokens) and the uninitialised ``empty_like`` output buffer is
   replaced by a ``-7`` fill.  Triton is re-run on the sanitised inputs and must
   reproduce the dump's ``out_after``.
2. ``syn:*`` -- synthetic gap fillers: draft probs (``NO_DRAFT_PROBS=False``),
   ``SYNTHETIC_MODE`` for greedy/random, fp64 inv_q, an inf-inv_q tie, ragged
   and empty requests, V in {1, 31, 1000}, and V=128256 with tiny batches.
   Greedy + ``SYNTHETIC_MODE`` + int64 drafts does not compile on triton-cpu
   (``inconsistent types int64 and int32``); the reference is Triton on the
   int32-cast drafts (``ref="triton_int32_cast"``).
3. ``e2e:*`` -- upstream ``rejection_sample()`` with its two random draws
   (``generate_uniform_probs`` and the exponential ``q`` in
   ``sample_recovered_tokens``) recorded and replayed
   (``_parity_common.recorded_randomness``), so the case is RNG-free.

Usage::

    TRITON_CPU_BACKEND=1 .venv_aot/bin/python tools/rs_parity/gen_golden_fixtures.py
    ... gen_golden_fixtures.py --check          # regenerate in memory, diff vs file
    ... gen_golden_fixtures.py --out /tmp/x.pt  # write elsewhere
    .venv_aot/bin/python tools/rs_parity/gen_golden_fixtures.py --update-hashes
        # refresh fixtures/upstream_hashes.json (tier C); does NOT need triton

``--check`` without the source dump directory re-verifies the stored dump
cases by re-running Triton on their stored inputs instead of re-curating.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("TRITON_CPU_BACKEND", "1")

REPO = Path(__file__).resolve().parents[2]
PARITY_DIR = REPO / "tests" / "unit" / "spec_decode" / "rejection_parity"
sys.path.insert(0, str(PARITY_DIR))

DEFAULT_DUMPS = REPO / "runs" / "identity_notriton" / "dumps"
SIZE_BUDGET = 3 * 1024 * 1024
PROD_VOCAB = 128256

GREEDY = "rejection_greedy_sample_kernel"
RANDOM = "rejection_random_sample_kernel"
RECOVERED = "sample_recovered_tokens_kernel"
EXPAND = "expand_kernel"


# --------------------------------------------------------------------------
# Triton reference
# --------------------------------------------------------------------------
def _triton_reference(common, name, grid, args, kwargs) -> tuple[Any, str]:
    """Run upstream Triton on a clone; returns (output, ref kind)."""
    import torch

    t_args = common.clone_args(args)
    try:
        common.launch_triton(name, grid, t_args, kwargs)
        return t_args[0], "triton"
    except Exception as exc:  # noqa: BLE001
        known = (
            name == GREEDY
            and kwargs.get("SYNTHETIC_MODE")
            and args[2].dtype == torch.int64
            and common.KNOWN_TRITON_SYNTHETIC_INT64 in str(exc)
        )
        if not known:
            raise
    t_args = common.clone_args(args)
    t_args[2] = t_args[2].to(torch.int32)  # ids < 2^31 -> identical values
    common.launch_triton(name, grid, t_args, kwargs)
    return t_args[0], "triton_int32_cast"


def _kernel_case(common, name, cname, grid, args, kwargs, source, tags):
    out, ref = _triton_reference(common, name, grid, args, kwargs)
    return {
        "name": cname,
        "kind": "kernel",
        "kernel": name,
        "grid": tuple(int(g) for g in grid),
        "args": common.clone_args(args),
        "kwargs": dict(kwargs),
        "output": out,
        "ref": ref,
        "source": source,
        "tags": sorted(tags),
    }


# --------------------------------------------------------------------------
# 1. curated real dumps
# --------------------------------------------------------------------------
def _dtype_sig(args) -> str:
    import torch

    return ",".join(
        "None"
        if a is None
        else (
            str(a.dtype).replace("torch.", "") if isinstance(a, torch.Tensor) else "s"
        )
        for a in args
    )


def _dump_key(d) -> tuple:
    mask = ""
    if d["kernel"] == GREEDY:
        mask = "none" if d["args"][5] is None else "tensor"
    kw = ",".join(f"{k}={v}" for k, v in sorted(d["kwargs"].items()))
    return (d["kernel"], kw, _dtype_sig(d["args"]), mask)


def _sanitize(name, args) -> list[str]:
    """Zero elements the kernel never reads (see module docstring)."""
    import torch

    notes = []
    cu = args[1]
    counts = torch.diff(torch.cat([torch.zeros(1, dtype=cu.dtype), cu])).long()
    if name == RECOVERED:
        args[0] = torch.full_like(args[0], -7)
        notes.append("output buffer (uninitialised empty_like) -> -7 fill")
        inv_q = args[5].clone()
        if (counts == 0).any():
            inv_q[counts == 0] = 0
            notes.append("inv_q rows of zero-draft requests zeroed (never read)")
        args[5] = inv_q
    elif name == RANDOM:
        draft = args[2].long()
        rows = torch.arange(draft.numel())
        for i in (3, 4):  # draft_probs, target_probs: only [t, draft[t]] is read
            if args[i] is None:
                continue
            p = torch.zeros_like(args[i])
            if draft.numel():
                p[rows, draft] = args[i][rows, draft]
            args[i] = p
            notes.append(f"arg{i} entries other than [t, draft[t]] zeroed (never read)")
    return notes


def curated_dump_cases(common, dumps: Path) -> list[dict[str, Any]]:
    import torch

    best: dict[tuple, tuple[int, str]] = {}
    files = sorted(dumps.rglob("*.pt"))
    if not files:
        raise SystemExit(f"no *.pt dumps under {dumps}")
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        key = _dump_key(d)
        cand = (f.stat().st_size, str(f.relative_to(dumps)))
        if key not in best or cand < best[key]:
            best[key] = cand
    cases = []
    for key in sorted(best):
        rel = best[key][1]
        d = torch.load(dumps / rel, map_location="cpu", weights_only=True)
        name, grid, kwargs = d["kernel"], tuple(d["grid"]), dict(d["kwargs"])
        args = list(d["args"])
        notes = _sanitize(name, args)
        case = _kernel_case(
            common,
            name,
            f"dump:{name.replace('_kernel', '')}:{'|'.join(key[1:])}",
            grid,
            args,
            kwargs,
            f"e2e_dump:{rel}",
            ["real"]
            + [f"{k}={int(v)}" for k, v in kwargs.items() if isinstance(v, bool)],
        )
        if not torch.equal(case["output"], d["out_after"]):
            raise SystemExit(
                f"{rel}: Triton on sanitised inputs != recorded out_after "
                f"({case['output'].tolist()} vs {d['out_after'].tolist()})"
            )
        case["dump_impl"] = d.get("impl", "?")
        case["sanitized"] = notes
        cases.append(case)
    return cases


# --------------------------------------------------------------------------
# 2. synthetic gap fillers
# --------------------------------------------------------------------------
def _ragged(g, batch, k):
    import torch

    counts = torch.randint(0, k + 1, (batch,), generator=g)
    if batch > 1:
        counts[0] = 0  # empty request first: start-offset logic
        counts[-1] = k
    return counts


def _probs(g, n, vocab, mode):
    import torch

    if mode == "peaked":
        p = (torch.randn(n, vocab, generator=g) * 4).softmax(-1, dtype=torch.float32)
    elif mode == "coarse":  # few distinct values: compresses well at V=128256
        logits = torch.randint(0, 4, (n, vocab), generator=g).float()
        p = logits.softmax(-1, dtype=torch.float32)
    else:  # ties
        p = (torch.rand(n, vocab, generator=g) * 4).floor() / 4
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
    return p.contiguous()


class _Syn:
    """Production-dtype synthetic batch (draft int64, cu int32, bonus int32)."""

    def __init__(self, seed, batch, k, vocab, probs="peaked", fp64=False, q="exp"):
        import torch

        g = torch.Generator().manual_seed(seed)
        self.g, self.batch, self.k, self.vocab = g, batch, k, vocab
        counts = _ragged(g, batch, k)
        self.counts = counts.tolist()
        self.cu = torch.cumsum(counts, 0).to(torch.int32)
        n = self.n = int(self.cu[-1])
        self.target = _probs(g, n, vocab, probs)
        self.argmax = self.target.argmax(-1)
        rnd = torch.randint(0, vocab, (n,), generator=g)
        if n:
            rnd[0], rnd[-1] = 0, vocab - 1
        accept = torch.rand(n, generator=g) < 0.6
        self.draft = torch.where(accept, self.argmax, rnd).contiguous()
        self.dprobs = _probs(g, n, vocab, probs)
        self.bonus = torch.randint(0, vocab, (batch, 1), generator=g, dtype=torch.int32)
        self.uniform = torch.rand(n, generator=g, dtype=torch.float64)
        self.is_greedy = torch.rand(batch, generator=g) < 0.5
        if batch > 1:
            self.is_greedy[0], self.is_greedy[-1] = True, False
        self.rates = torch.rand(k, generator=g).to(torch.float32)
        qq = torch.empty(batch, vocab, dtype=torch.float64 if fp64 else torch.float32)
        qq.exponential_(generator=g)
        if q == "int":
            qq = qq.round().clamp_min(1)
        elif q == "large_inv":
            # Tiny q on ~1% of entries -> inv_q up to ~1e300 (fp64), which an
            # fp32 score computation would overflow; catches lost fp64 precision.
            tiny = 1e-300 if fp64 else 1e-30
            mask = torch.rand(batch, vocab, generator=g) < 0.01
            qq = torch.where(mask, qq * tiny, qq)
        self.inv_q = qq.reciprocal()
        self.fp64 = fp64

    def out(self):
        import torch

        return torch.full((self.batch, self.k + 1), -1, dtype=torch.int32)

    def zero_unused_q_rows(self):
        import torch

        counts = torch.tensor(self.counts)
        self.inv_q[counts == 0] = 0

    def recovered(self, no_draft):
        import torch

        args = [
            torch.full_like(self.draft, -7),
            self.cu,
            self.draft,
            None if no_draft else self.dprobs,
            self.target,
            self.inv_q,
            self.vocab,
            8192,
        ]
        kw = dict(NO_DRAFT_PROBS=no_draft, USE_FP64_GUMBEL=self.fp64)
        return (self.batch, self.k), args, kw

    def greedy(self, mask, synthetic):
        uniform = self.uniform if (synthetic or mask is not None) else None
        args = [
            self.out(),
            self.cu,
            self.draft,
            self.argmax,
            self.bonus,
            self.is_greedy if mask else None,
            self.k,
            uniform,
            self.rates if synthetic else None,
        ]
        return (self.batch,), args, dict(SYNTHETIC_MODE=synthetic)

    def random(self, no_draft, synthetic, recovered):
        args = [
            self.out(),
            self.cu,
            self.draft,
            None if no_draft else self.dprobs,
            self.target,
            self.bonus,
            recovered,
            self.uniform,
            self.is_greedy,
            self.k,
            self.vocab,
            self.rates if synthetic else None,
        ]
        return (
            (self.batch,),
            args,
            dict(NO_DRAFT_PROBS=no_draft, SYNTHETIC_MODE=synthetic),
        )

    def adversarial_random(self, no_draft, synthetic):
        """uniform == accept ratio (>= accepts), == rate (< rejects), dprob == 0."""
        import torch

        n = self.n
        if not n:
            return
        idx = torch.arange(n)
        sel = idx[idx % 3 == 0]
        d = self.draft[sel]
        tp = self.target[sel, d]
        ratio = tp if no_draft else tp / self.dprobs[sel, d]
        self.uniform[sel] = ratio.to(torch.float64)
        if not no_draft:
            z = idx[idx % 5 == 1]
            self.dprobs[z, self.draft[z]] = 0.0
        if synthetic:
            starts = torch.cat([torch.zeros(1, dtype=torch.int64), self.cu[:-1].long()])
            pos = idx - torch.repeat_interleave(starts, torch.tensor(self.counts))
            hit = idx[idx % 4 == 2]
            self.uniform[hit] = self.rates[pos[hit]].to(torch.float64)


def synthetic_cases(common) -> list[dict[str, Any]]:
    import torch

    cases: list[dict[str, Any]] = []

    def add(name, kernel, launch, tags):
        grid, args, kw = launch
        cases.append(
            _kernel_case(
                common, kernel, f"syn:{name}", grid, args, kw, "synthetic", tags
            )
        )

    # recovered: NO_DRAFT_PROBS x fp64 x V
    seed = 100
    for vocab, batch, k in ((1, 3, 2), (31, 4, 3), (1000, 3, 3)):
        for no_draft in (True, False):
            for fp64 in (False, True):
                seed += 1
                c = _Syn(seed, batch, k, vocab, fp64=fp64)
                c.zero_unused_q_rows()
                add(
                    f"recovered:V{vocab}:nodraft{int(no_draft)}:fp64{int(fp64)}",
                    RECOVERED,
                    c.recovered(no_draft),
                    [
                        f"V={vocab}",
                        f"NO_DRAFT_PROBS={int(no_draft)}",
                        f"USE_FP64_GUMBEL={int(fp64)}",
                        "ragged",
                    ],
                )
    # recovered: integer q -> exact score ties; inf inv_q tie (first index wins)
    c = _Syn(150, 3, 3, 1000, probs="ties", q="int")
    add(
        "recovered:V1000:score_ties",
        RECOVERED,
        c.recovered(False),
        ["V=1000", "NO_DRAFT_PROBS=0", "ties"],
    )
    for no_draft in (True, False):
        c = _Syn(151, 2, 3, 1000)
        c.inv_q[:, [17, 400, 999]] = float("inf")
        add(
            f"recovered:V1000:inf_inv_q_tie:nodraft{int(no_draft)}",
            RECOVERED,
            c.recovered(no_draft),
            ["V=1000", f"NO_DRAFT_PROBS={int(no_draft)}", "inf_inv_q"],
        )

    # recovered: fp64 inv_q beyond fp32 range (tier-A q_mode="large_inv")
    for no_draft in (True, False):
        c = _Syn(152 + int(no_draft), 4, 3, 1000, fp64=True, q="large_inv")
        c.zero_unused_q_rows()
        add(
            f"recovered:V1000:large_inv:fp641:nodraft{int(no_draft)}",
            RECOVERED,
            c.recovered(no_draft),
            [
                "V=1000",
                f"NO_DRAFT_PROBS={int(no_draft)}",
                "USE_FP64_GUMBEL=1",
                "large_inv",
            ],
        )

    # greedy: mask x SYNTHETIC_MODE (int64 drafts) + V=1 + prod-vocab ids
    for mask in (False, True):
        for synthetic in (False, True):
            c = _Syn(200 + 2 * mask + synthetic, 6, 4, 1000)
            if synthetic and c.n:
                c.uniform[0] = float(c.rates[0])  # uniform == rate -> reject
            add(
                f"greedy:V1000:mask{int(mask)}:syn{int(synthetic)}",
                GREEDY,
                c.greedy(mask, synthetic),
                [
                    "V=1000",
                    f"mask={'tensor' if mask else 'none'}",
                    f"SYNTHETIC_MODE={int(synthetic)}",
                ],
            )
    c = _Syn(210, 3, 2, 1)
    add("greedy:V1:mask1:syn0", GREEDY, c.greedy(True, False), ["V=1"])
    c = _Syn(211, 2, 4, 64)
    c.argmax = torch.randint(0, PROD_VOCAB, (c.n,), generator=c.g)
    c.draft = torch.where(
        torch.rand(c.n, generator=c.g) < 0.7,
        c.argmax,
        torch.randint(0, PROD_VOCAB, (c.n,), generator=c.g),
    )
    c.bonus = torch.randint(0, PROD_VOCAB, (2, 1), generator=c.g, dtype=torch.int32)
    add(
        f"greedy:V{PROD_VOCAB}ids:mask0:syn1",
        GREEDY,
        c.greedy(False, True),
        [f"V={PROD_VOCAB}", "SYNTHETIC_MODE=1"],
    )

    # random: NO_DRAFT_PROBS x SYNTHETIC_MODE, adversarial boundaries
    seed = 300
    for vocab in (31, 1000):
        for no_draft in (True, False):
            for synthetic in (False, True):
                seed += 1
                c = _Syn(seed, 4, 3, vocab)
                c.adversarial_random(no_draft, synthetic)
                g, a, kw = c.recovered(no_draft)
                common.launch_triton(RECOVERED, g, a, kw)
                add(
                    f"random:V{vocab}:nodraft{int(no_draft)}:syn{int(synthetic)}",
                    RANDOM,
                    c.random(no_draft, synthetic, a[0]),
                    [
                        f"V={vocab}",
                        f"NO_DRAFT_PROBS={int(no_draft)}",
                        f"SYNTHETIC_MODE={int(synthetic)}",
                        "adversarial",
                    ],
                )
    c = _Syn(320, 3, 2, 1)
    add(
        "random:V1:nodraft0:syn0",
        RANDOM,
        c.random(False, False, torch.zeros_like(c.draft)),
        ["V=1"],
    )

    # V=128256 tiny batches (coarse probs / integer q compress well)
    c = _Syn(400, 1, 2, PROD_VOCAB, probs="coarse", q="int")
    add(
        f"recovered:V{PROD_VOCAB}:nodraft0:fp64_0",
        RECOVERED,
        c.recovered(False),
        [f"V={PROD_VOCAB}", "NO_DRAFT_PROBS=0", "USE_FP64_GUMBEL=0"],
    )
    c = _Syn(401, 2, 1, PROD_VOCAB, probs="coarse", fp64=True, q="int")
    c.zero_unused_q_rows()
    add(
        f"recovered:V{PROD_VOCAB}:nodraft1:fp64_1",
        RECOVERED,
        c.recovered(True),
        [f"V={PROD_VOCAB}", "NO_DRAFT_PROBS=1", "USE_FP64_GUMBEL=1"],
    )
    c = _Syn(402, 2, 2, PROD_VOCAB, probs="coarse")
    c.is_greedy[:] = False
    c.adversarial_random(False, False)
    grid, rargs, kw = c.random(
        False, False, torch.randint(0, PROD_VOCAB, (c.n,), generator=c.g)
    )
    _sanitize(RANDOM, rargs)  # only [t, draft[t]] is read; keeps the file small
    add(
        f"random:V{PROD_VOCAB}:nodraft0:syn0",
        RANDOM,
        (grid, rargs, kw),
        [f"V={PROD_VOCAB}", "NO_DRAFT_PROBS=0", "adversarial", "sanitized"],
    )

    # expand: the three apply_sampling_constraints / expand_batch_to_tokens sites
    from vllm.v1.sample import rejection_sampler as rs

    pools: dict[str, tuple[Any, list[float], float, int]] = {
        "temperature": (
            torch.float32,
            [0.0, 0.7, 1.0, 1e-5, 2.0],
            rs.GREEDY_TEMPERATURE,
            1,
        ),
        "top_k": (torch.int32, [0, 1, 50, PROD_VOCAB, 2**31 - 1], 0, 0),
        "top_p": (torch.float32, [1.0, 0.9, 0.0, 1e-7, 0.95], 0, 0),
    }
    for i, (param, (dtype, pool, replace_from, replace_to)) in enumerate(
        sorted(pools.items())
    ):
        g = torch.Generator().manual_seed(500 + i)
        counts = _ragged(g, 7, 4)
        cu = torch.cumsum(counts, 0).to(torch.int32)
        pool_t = torch.tensor(pool, dtype=dtype)
        x = pool_t[torch.randint(0, len(pool), (7,), generator=g)].contiguous()
        out = x.new_full((int(cu[-1]),), -3)
        add(
            f"expand:{param}",
            EXPAND,
            (
                (7,),
                [out, x, cu, replace_from, replace_to],
                dict(MAX_NUM_TOKENS=rs.MAX_SPEC_LEN),
            ),
            [f"x={str(dtype).replace('torch.', '')}", "ragged"],
        )
    return cases


# --------------------------------------------------------------------------
# 3. RNG-free end-to-end rejection_sample()
# --------------------------------------------------------------------------
def e2e_cases(common) -> list[dict[str, Any]]:
    import torch

    from vllm.v1.sample import rejection_sampler as rs

    specs = [  # (name, mode, synthetic, fp64, draft_probs)
        ("greedy", "greedy", False, False, False),
        ("mixed", "mixed", False, False, False),
        ("random", "random", False, False, False),
        ("random_draftprobs_fp64", "random", False, True, True),
        ("mixed_synthetic", "mixed", True, False, False),
    ]
    cases = []
    for i, (name, mode, synthetic, fp64, use_dprobs) in enumerate(specs):
        g = torch.Generator().manual_seed(600 + i)
        batch, k, vocab = 4, 4, 256
        counts = _ragged(g, batch, k)
        cu = torch.cumsum(counts, 0).to(torch.int32)
        n = int(cu[-1])
        logits = torch.randn(n, vocab, generator=g) * 3
        argmax = logits.argmax(-1)
        draft = torch.where(
            torch.rand(n, generator=g) < 0.6,
            argmax,
            torch.randint(0, vocab, (n,), generator=g),
        ).contiguous()
        temp = torch.full((batch,), 0.7, dtype=torch.float32)
        if mode == "greedy":
            temp.zero_()
        elif mode == "mixed":
            temp[torch.tensor([True, False] * (batch // 2))] = 0.0
        dprobs = _probs(g, n, vocab, "peaked") if use_dprobs else None
        need_uniform = synthetic or mode != "greedy"
        need_q = mode != "greedy"
        q = torch.empty(batch, vocab, dtype=torch.float64 if fp64 else torch.float32)
        q.exponential_(generator=g)
        inputs = {
            "draft_token_ids": draft,
            "num_draft_tokens": [int(c) for c in counts],
            "max_spec_len": k,
            "cu_num_draft_tokens": cu,
            "draft_probs": dprobs,
            "target_logits": logits,
            "bonus_token_ids": torch.randint(
                0, vocab, (batch, 1), generator=g, dtype=torch.int32
            ),
            "temperature": temp,
            "all_greedy": mode == "greedy",
            "all_random": mode == "random",
            "synthetic_mode": synthetic,
            "synthetic_conditional_rates": torch.rand(k, generator=g)
            if synthetic
            else None,
            "use_fp64_gumbel": fp64,
            "uniform": torch.rand(n, generator=g, dtype=torch.float64)
            if need_uniform
            else None,
            "inv_q": q.reciprocal() if need_q else None,
        }
        saved = {nm: getattr(rs, nm) for nm in common.KERNEL_NAMES}
        try:
            for nm in common.KERNEL_NAMES:
                setattr(rs, nm, common.ORIGINAL_KERNELS[nm])
            ref = "triton"
            try:
                out = common.run_rejection_sample(inputs)
            except Exception as exc:  # noqa: BLE001
                if not (synthetic and common.KNOWN_TRITON_SYNTHETIC_INT64 in str(exc)):
                    raise
                out = common.run_rejection_sample(inputs, draft.to(torch.int32))
                ref = "triton_int32_cast"
        finally:
            for nm, obj in saved.items():
                setattr(rs, nm, obj)
        cases.append(
            {
                "name": f"e2e:{name}",
                "kind": "rejection_sample",
                "kernel": "rejection_sample",
                "inputs": inputs,
                "output": out,
                "ref": ref,
                "source": "synthetic",
                "tags": sorted(
                    [
                        f"mode={mode}",
                        f"SYNTHETIC_MODE={int(synthetic)}",
                        f"USE_FP64_GUMBEL={int(fp64)}",
                        f"NO_DRAFT_PROBS={int(not use_dprobs)}",
                        f"V={vocab}",
                    ]
                ),
            }
        )
    return cases


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------
def _deflate_torch_save(obj: Any, path: Path) -> None:
    import torch

    buf = io.BytesIO()
    # The established fixture format is a trusted internal artifact. Its
    # payload is tensors plus primitive/container metadata and is read with
    # weights_only=True above and by the parity tests.
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save(obj, buf)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zin, zipfile.ZipFile(path, "w") as zout:
        for info in zin.infolist():
            zi = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(zi, zin.read(info.filename), compresslevel=9)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _inventory(cases) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{c['kernel']}|{c['source'].split(':')[0]}|ref={c['ref']}"
                for c in cases
            ).items()
        )
    )


def stored_dump_cases(common, pt: Path) -> list[dict[str, Any]]:
    """Dump cases from the committed fixture, with fresh Triton outputs.

    Used when the raw E2E dumps (an untracked ``runs/`` artifact) are absent:
    the sanitised inputs are already stored, so only the reference is redone.
    """
    if not pt.is_file():
        raise SystemExit(f"neither dumps nor a stored fixture ({pt}) to reuse")
    cases = []
    for c in common.load_golden(str(pt))["cases"].values():
        if not c["source"].startswith("e2e_dump:"):
            continue
        case = _kernel_case(
            common,
            c["kernel"],
            c["name"],
            tuple(c["grid"]),
            list(c["args"]),
            dict(c["kwargs"]),
            c["source"],
            list(c["tags"]),
        )
        for k in ("dump_impl", "sanitized"):
            if k in c:
                case[k] = c[k]
        cases.append(case)
    return cases


def _dump_source(common, dumps: Path, pt: Path | None) -> str:
    # Reused stored dump cases keep the provenance of the original dumps.
    if not dumps.is_dir() and pt is not None and pt.is_file():
        meta = json.loads(Path(common.golden_json_path(str(pt))).read_text())
        return meta["dump_source"]
    return str(dumps.relative_to(REPO)) if dumps.is_relative_to(REPO) else str(dumps)


def build(
    common, dumps: Path, pt: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    import triton

    if dumps.is_dir():
        dump_cases = curated_dump_cases(common, dumps)
    else:
        if pt is None:
            raise SystemExit(f"no dumps at {dumps}")
        print(f"NOTE: {dumps} missing; reusing dump-case inputs stored in {pt}")
        dump_cases = stored_dump_cases(common, pt)
    cases = dump_cases + synthetic_cases(common) + e2e_cases(common)
    names = [c["name"] for c in cases]
    dup = [n for n, k in Counter(names).items() if k > 1]
    if dup:
        raise SystemExit(f"duplicate case names: {dup}")
    for c in cases:
        c["sha256"] = common.case_sha256(c)
    payload = {"format": common.GOLDEN_FORMAT, "cases": {c["name"]: c for c in cases}}
    meta = {
        "format": common.GOLDEN_FORMAT,
        "versions": {
            "triton": getattr(triton, "__version__", "?"),
            "triton_dist": version("triton"),
            "vllm": version("vllm"),
            "torch": version("torch"),
            "numba": version("numba"),
            "llvmlite": version("llvmlite"),
        },
        "generator_git_sha": _git_sha(),
        "dump_source": _dump_source(common, dumps, pt),
        "num_cases": len(cases),
        "inventory": _inventory(cases),
        "cases": [
            {
                k: c[k]
                for k in ("name", "kind", "kernel", "ref", "source", "tags", "sha256")
            }
            | (
                {"grid": list(c["grid"]), "kwargs": c["kwargs"]}
                if c["kind"] == "kernel"
                else {}
            )
            | (
                {"dump_impl": c["dump_impl"], "sanitized": c["sanitized"]}
                if "dump_impl" in c
                else {}
            )
            for c in cases
        ],
    }
    return payload, meta


def _require_triton(common) -> None:
    from _triton_probe import triton_cpu_status

    ok, why = triton_cpu_status()
    if not ok:
        raise SystemExit(f"refusing to generate golden fixtures: {why}")
    common.assert_triton_originals()


def check(common, pt: Path, dumps: Path) -> int:
    import torch

    stored = common.load_golden(str(pt))["cases"]
    diffs = []
    fresh, _ = build(common, dumps, pt)
    fresh_cases = fresh["cases"]
    for name in sorted(set(stored) | set(fresh_cases)):
        if name not in stored:
            diffs.append(f"{name}: new case (not in {pt.name})")
        elif name not in fresh_cases:
            diffs.append(f"{name}: stored case no longer generated")
        elif stored[name]["sha256"] != fresh_cases[name]["sha256"]:
            out_eq = torch.equal(stored[name]["output"], fresh_cases[name]["output"])
            what = "same" if out_eq else "DIFFERS"
            diffs.append(f"{name}: content sha differs (output {what})")
    for d in diffs:
        print("DIFF", d)
    print(
        f"--check: {len(stored)} stored, {len(fresh_cases)} regenerated, "
        f"{len(diffs)} differences"
    )
    return 1 if diffs else 0


def main() -> int:
    import _parity_common as common

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path(common.GOLDEN_PT))
    ap.add_argument("--dumps", type=Path, default=DEFAULT_DUMPS)
    ap.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and compare with --out",
    )
    ap.add_argument(
        "--update-hashes",
        action="store_true",
        help="refresh fixtures/upstream_hashes.json only (no triton needed)",
    )
    args = ap.parse_args()

    if args.update_hashes:
        import _upstream_sources as us

        rec = us.write_recorded()
        print(
            f"wrote {us.HASHES_PATH} (vllm {rec['vllm_version']}): "
            f"{len(rec['hashes'])} hashes, constants={rec['constants']}"
        )
        return 0

    _require_triton(common)
    if args.check:
        return check(common, args.out, args.dumps)

    payload, meta = build(common, args.dumps, args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _deflate_torch_save(payload, args.out)
    meta["pt_sha256"] = common.file_sha256(str(args.out))
    jpath = Path(common.golden_json_path(str(args.out)))
    jpath.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    total = args.out.stat().st_size + jpath.stat().st_size
    print(f"wrote {args.out} + {jpath.name}: {meta['num_cases']} cases, {total} bytes")
    for k, v in meta["inventory"].items():
        print(f"  {v:3d}  {k}")
    if total > SIZE_BUDGET:
        print(f"ERROR: fixtures exceed the {SIZE_BUDGET}-byte budget")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

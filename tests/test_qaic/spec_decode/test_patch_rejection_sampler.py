# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""CPU-only tests for the QAIC ``RejectionSampler.forward`` patch.

The patched forward must be output-equivalent to upstream vLLM's forward; its
only intended delta is skipping a tensor clone on the greedy/no-logprobs path.
Upstream Triton kernels run on CPU through triton-cpu::

    TRITON_CPU_BACKEND=1 .venv_aot/bin/python -m pytest -s \
        tests/test_qaic/spec_decode/test_patch_rejection_sampler.py -v
"""

import importlib.util
import os
import sys

os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import pytest
import torch

import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler
import vllm.v1.sample.rejection_sampler as upstream_rs
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_qaic.patch.patch_rejection_sampler as qaic_patch
from vllm_qaic.v1.sample import topk_topp_sampler_shim


def _has_triton_cpu_backend() -> bool:
    try:
        from triton.backends import backends

        return "cpu" in backends
    except Exception:
        return False


# Upstream rejection_sample launches Triton kernels on CPU tensors.
requires_triton_cpu = pytest.mark.skipif(
    not _has_triton_cpu_backend(),
    reason="Requires the triton-cpu backend to run upstream kernels on CPU",
)

VOCAB_SIZE = 64
DRAFT_TOKEN_IDS = [[3, 7, 11], [5, 9], [2, 4, 6, 8]]


def _load_pristine_upstream_module():
    """Load a private copy of upstream rejection_sampler.py.

    ``vllm_qaic.patch`` overwrites ``RejectionSampler.forward`` in place without
    keeping the original, so re-execute the upstream source under a private
    module name to obtain the unpatched ``forward``.
    """
    name = "_pristine_vllm_rejection_sampler"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, upstream_rs.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_UPSTREAM_FORWARD = _load_pristine_upstream_module().RejectionSampler.forward


def _make_spec_metadata(draft_token_ids: list[list[int]]) -> SpecDecodeMetadata:
    metadata = SpecDecodeMetadata.make_dummy(draft_token_ids, torch.device("cpu"))
    # make_dummy zero-fills the indices; build the real flattened layout
    # [d0_0, ..., d0_n, bonus0, d1_0, ..., bonus1, ...].
    target, bonus, offset = [], [], 0
    for ids in draft_token_ids:
        target.extend(range(offset, offset + len(ids)))
        bonus.append(offset + len(ids))
        offset += len(ids) + 1
    metadata.target_logits_indices = torch.tensor(target, dtype=torch.int32)
    metadata.bonus_logits_indices = torch.tensor(bonus, dtype=torch.int32)
    metadata.logits_indices = torch.arange(offset, dtype=torch.int32)
    return metadata


def _make_sampling_metadata(
    temperatures: list[float],
    seed: int,
    top_k: list[int] | None = None,
    top_p: list[float] | None = None,
    max_num_logprobs: int | None = None,
) -> SamplingMetadata:
    all_greedy = all(t == 0.0 for t in temperatures)
    all_random = all(t != 0.0 for t in temperatures)
    generators = {
        i: torch.Generator().manual_seed(seed + i) for i in range(len(temperatures))
    }
    return SamplingMetadata(
        temperature=None if all_greedy else torch.tensor(temperatures),
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=None if top_p is None else torch.tensor(top_p),
        top_k=None if top_k is None else torch.tensor(top_k, dtype=torch.int32),
        generators=generators,
        max_num_logprobs=max_num_logprobs,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([]),
        presence_penalties=torch.tensor([]),
        repetition_penalties=torch.tensor([]),
        output_token_ids=[[] for _ in temperatures],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _make_rejection_sampler(
    use_fp64_gumbel: bool = False, synthetic_rates: list[float] | None = None
) -> RejectionSampler:
    sampler = RejectionSampler(Sampler(use_fp64_gumbel=use_fp64_gumbel))
    if synthetic_rates is not None:
        # Equivalent to spec_config.rejection_sample_method == "synthetic".
        sampler.synthetic_conditional_rates = torch.tensor(
            synthetic_rates, dtype=torch.float32
        )
        sampler.synthetic_mode = True
    return sampler


def _install_topk_topp_shim(monkeypatch) -> None:
    """Install the top-k/top-p shim reversibly, as QAIC workers do.

    Without it, >= 8-row top-k/top-p on the AOT platform selects the Triton
    kernel, which calls the unavailable ``num_compute_units``.
    """
    monkeypatch.setattr(
        topk_topp_sampler,
        "apply_top_k_top_p_triton",
        topk_topp_sampler.apply_top_k_top_p_triton,
    )
    monkeypatch.setattr(topk_topp_sampler_shim, "_shim_installed", False)
    topk_topp_sampler_shim.install()


def _run(forward, sampler, metadata, logits, draft_probs, sampling_kwargs):
    torch.manual_seed(1234)
    return forward(
        sampler,
        metadata,
        None if draft_probs is None else draft_probs.clone(),
        logits.clone(),
        _make_sampling_metadata(**sampling_kwargs),
    )


CASES = {
    "all_greedy": dict(temperatures=[0.0, 0.0, 0.0]),
    "all_greedy_logprobs": dict(temperatures=[0.0, 0.0, 0.0], max_num_logprobs=2),
    "all_random": dict(temperatures=[0.7, 0.7, 0.7]),
    "mixed": dict(temperatures=[0.0, 0.7, 1.3]),
    "mixed_logprobs": dict(temperatures=[0.0, 0.7, 1.3], max_num_logprobs=0),
    "top_k_top_p": dict(
        temperatures=[0.7, 0.9, 1.1], top_k=[5, 10, 64], top_p=[0.8, 0.95, 1.0]
    ),
}


@requires_triton_cpu
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("with_draft_probs", [False, True])
@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("use_fp64_gumbel", [False, True])
@pytest.mark.parametrize("case", list(CASES))
def test_qaic_forward_matches_upstream(
    case, use_fp64_gumbel, synthetic, with_draft_probs, seed, monkeypatch
):
    _install_topk_topp_shim(monkeypatch)
    assert RejectionSampler.forward is qaic_patch._qaic_forward
    metadata = _make_spec_metadata(DRAFT_TOKEN_IDS)
    num_tokens = sum(len(ids) for ids in DRAFT_TOKEN_IDS)
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_tokens + len(DRAFT_TOKEN_IDS), VOCAB_SIZE, generator=g)
    draft_probs = None
    if with_draft_probs:
        draft_probs = torch.rand(num_tokens, VOCAB_SIZE, generator=g)
        draft_probs = (draft_probs / draft_probs.sum(-1, keepdim=True)).contiguous()
    sampling_kwargs = dict(CASES[case], seed=100 + seed)
    rates = [0.9, 0.5, 0.2, 0.1] if synthetic else None

    sampler = _make_rejection_sampler(use_fp64_gumbel, rates)
    expected = _run(
        _UPSTREAM_FORWARD, sampler, metadata, logits, draft_probs, sampling_kwargs
    )
    actual = _run(
        qaic_patch._qaic_forward,
        sampler,
        metadata,
        logits,
        draft_probs,
        sampling_kwargs,
    )

    assert torch.equal(actual.sampled_token_ids, expected.sampled_token_ids)
    if expected.logprobs_tensors is None:
        assert actual.logprobs_tensors is None
    else:
        for name in ("logprob_token_ids", "logprobs", "selected_token_ranks"):
            assert torch.equal(
                getattr(actual.logprobs_tensors, name),
                getattr(expected.logprobs_tensors, name),
            ), name


@requires_triton_cpu
def test_rejection_sample_receives_constrained_logits_and_options(monkeypatch):
    """Regression: the patch used to pass softmax(logits) (upstream softmaxes
    again -> double softmax) and dropped the synthetic/fp64 options."""
    calls = []
    real_rejection_sample = qaic_patch.rejection_sample

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_rejection_sample(*args, **kwargs)

    monkeypatch.setattr(qaic_patch, "rejection_sample", spy)

    temperatures = [0.5, 0.7, 2.0]
    metadata = _make_spec_metadata(DRAFT_TOKEN_IDS)
    num_tokens = sum(len(ids) for ids in DRAFT_TOKEN_IDS)
    logits = torch.randn(
        num_tokens + len(DRAFT_TOKEN_IDS),
        VOCAB_SIZE,
        generator=torch.Generator().manual_seed(0),
    )
    sampler = _make_rejection_sampler(
        use_fp64_gumbel=True, synthetic_rates=[0.9, 0.5, 0.2, 0.1]
    )
    sampler(
        metadata,
        None,
        logits.clone(),
        _make_sampling_metadata(temperatures, seed=0),
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    per_token_temperature = torch.tensor(temperatures).repeat_interleave(
        torch.tensor([len(ids) for ids in DRAFT_TOKEN_IDS])
    )
    expected_logits = logits[
        metadata.target_logits_indices.long()
    ] / per_token_temperature.unsqueeze(-1)
    torch.testing.assert_close(args[5], expected_logits)
    # Logits, not a probability distribution.
    assert not torch.allclose(args[5].sum(-1), torch.ones(num_tokens))
    assert kwargs["synthetic_mode"] is True
    assert kwargs["synthetic_conditional_rates"] is sampler.synthetic_conditional_rates
    assert kwargs["use_fp64_gumbel"] is True


def test_topk_topp_shim_avoids_num_compute_units(monkeypatch):
    from vllm.platforms import current_platform
    from vllm.triton_utils import HAS_TRITON

    def _unavailable(cls, device_id: int = 0) -> int:
        raise NotImplementedError("num_compute_units is not available in AOT mode.")

    monkeypatch.setattr(
        type(current_platform), "num_compute_units", classmethod(_unavailable)
    )
    # Snapshot the unshimmed state so the shim is undone after the test.
    monkeypatch.setattr(
        topk_topp_sampler,
        "apply_top_k_top_p_triton",
        topk_topp_sampler.apply_top_k_top_p_triton,
    )
    monkeypatch.setattr(topk_topp_sampler_shim, "_shim_installed", False)

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(16, VOCAB_SIZE, generator=g)
    k = torch.full((16,), 5, dtype=torch.int32)
    p = torch.full((16,), 0.9)

    if HAS_TRITON and not current_platform.is_cpu():
        # Control: without the shim, >= 8 rows select the Triton kernel, which
        # sizes its grid with num_compute_units.
        with pytest.raises(NotImplementedError):
            topk_topp_sampler.apply_top_k_top_p(logits.clone(), k, p)

    topk_topp_sampler_shim.install()
    assert topk_topp_sampler_shim._shim_installed
    out = topk_topp_sampler.apply_top_k_top_p(logits.clone(), k, p)
    expected = topk_topp_sampler.apply_top_k_top_p_pytorch(logits.clone(), k, p)
    assert torch.equal(out, expected)
    assert torch.isfinite(out).sum(-1).le(5).all()

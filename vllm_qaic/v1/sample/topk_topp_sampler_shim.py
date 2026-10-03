# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/sample/ops/topk_topp_sampler.py

"""QAIC top-k/top-p sampler shim (AOT).

vLLM selects a Triton top-k/top-p kernel whenever ``HAS_TRITON`` is true and the
batch size is ``>= 8`` (``vllm/v1/sample/ops/topk_topp_sampler.py:345``). In AOT
mode, sampling runs on host CPU tensors, but the platform is QAIC, not CPU, so
the selector takes the batch-size branch. When a Triton build is importable
(e.g. triton-cpu), ``apply_top_k_top_p_triton`` sizes its grid with
``current_platform.num_compute_units()``, which raises ``NotImplementedError``
in AOT mode (``vllm_qaic/platform_base.py``).

That failure aborts engine init during the model profile-run warmup (which
builds a dummy random ``SamplingMetadata`` with ``top_p``/``top_k`` at
``max_num_seqs`` requests), independent of whether real requests use greedy
decoding.

The shim redirects the Triton entry point to the pure-PyTorch reference
implementation (``apply_top_k_top_p_pytorch``). We patch the leaf symbol
``apply_top_k_top_p_triton`` rather than the ``apply_top_k_top_p`` selector: the
selector resolves ``apply_top_k_top_p_triton`` from its defining module's globals
at call time, so patching the leaf redirects every caller (the sampler's
``forward_native`` and the ``sampler.py``/``states.py`` importers of
``apply_top_k_top_p``) uniformly, on both the CPU and the batch-size branches.

The shim only depends on upstream vLLM (no ``torch_qaic``/Hexagon imports).
"""

from __future__ import annotations

from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

_shim_installed = False


def install() -> None:
    """Route QAIC top-k/top-p sampling onto the PyTorch reference path.

    Idempotent: safe to call multiple times within one process.
    """
    global _shim_installed
    if _shim_installed:
        return

    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    def _qaic_apply_top_k_top_p_pytorch(logits, k, p):
        # Match apply_top_k_top_p_triton's (logits, k, p) call signature; the
        # Triton kernel is unusable on QAIC AOT (num_compute_units is
        # unavailable), so always take the PyTorch path.
        return apply_top_k_top_p_pytorch(logits, k, p)

    topk_topp_sampler.apply_top_k_top_p_triton = _qaic_apply_top_k_top_p_pytorch

    _shim_installed = True
    logger.info(
        "vllm_qaic: top-k/top-p sampling routed to PyTorch on QAIC "
        "(num_compute_units is unavailable in AOT mode)."
    )

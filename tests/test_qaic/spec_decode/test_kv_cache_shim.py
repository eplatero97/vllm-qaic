# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from types import SimpleNamespace

import torch

from vllm_qaic.v1.worker.kv_cache_shim import bind_kv_cache


def test_bind_kv_cache_accepts_target_and_draft_layer_indices():
    target_cache = torch.empty(1)
    draft_cache = torch.empty(1)
    kv_caches = {
        "model.layers.0.self_attn.attn": target_cache,
        "draft_model.model.layers.0.self_attn.attn": draft_cache,
    }
    forward_context = {
        layer_name: SimpleNamespace(kv_cache=None) for layer_name in kv_caches
    }
    runner_kv_caches = []

    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)

    assert runner_kv_caches[0] is target_cache
    assert runner_kv_caches[1] is draft_cache
    assert forward_context["model.layers.0.self_attn.attn"].kv_cache is target_cache
    assert (
        forward_context["draft_model.model.layers.0.self_attn.attn"].kv_cache
        is draft_cache
    )

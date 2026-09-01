# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""QAIC PYT KV-cache binding compatibility for model-based speculation."""

from collections import defaultdict

import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.utils import extract_layer_index
from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

_shim_installed = False


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """Bind target and draft KV caches for the QAIC eager runner.

    Model-based speculative decoding gives target and draft attention layers
    the same numeric layer indices. Upstream permits that layout for GPU and
    CPU runners but rejects other platforms before performing the otherwise
    platform-independent binding. QAIC uses the same flat cache-list contract,
    so retain every cache in layer-index order and bind it to its attention
    layer.
    """
    assert not runner_kv_caches

    index_to_names: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        layer_index = extract_layer_index(layer_name, num_attn_module)
        index_to_names[layer_index].append(layer_name)

    for layer_index in sorted(index_to_names):
        for layer_name in index_to_names[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


def install() -> None:
    """Install the QAIC binder into the inherited vLLM V1 model runner."""
    global _shim_installed
    if _shim_installed:
        return

    import vllm.v1.worker.gpu_model_runner as gpu_model_runner

    gpu_model_runner.bind_kv_cache = bind_kv_cache
    _shim_installed = True
    logger.info("vllm_qaic: QAIC PYT model-based SpD KV-cache binder installed.")

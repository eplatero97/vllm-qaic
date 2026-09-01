# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""QAIC PYT replacements for unsupported speculative-input kernels."""

from typing import Any

import torch

from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class _GridLaunchable:
    def __init__(self, fn):
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", repr(fn))

    def __getitem__(self, _grid: Any):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _copy_to_device(destination: torch.Tensor, source: torch.Tensor) -> None:
    destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def _prepare_next_token_ids_padded(
    sampled_token_ids: torch.Tensor,
    discard_request_mask: torch.Tensor,
    backup_next_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    vocab_size: int,
    num_sampled_tokens_per_req: int,
    num_reqs: int,
    stride_sampled_token_ids: int,
    BLOCK_SIZE_TOKENS: int,
) -> None:
    del stride_sampled_token_ids, BLOCK_SIZE_TOKENS

    sampled = sampled_token_ids[:num_reqs, :num_sampled_tokens_per_req].cpu()
    discarded = discard_request_mask[:num_reqs].cpu()
    backups = backup_next_token_ids[:num_reqs].cpu()
    valid_mask = (sampled != -1) & (sampled < vocab_size)
    valid_counts = valid_mask.sum(dim=1, dtype=torch.int32)

    token_offsets = torch.arange(
        num_sampled_tokens_per_req,
        dtype=torch.int32,
    ).unsqueeze(0)
    last_valid_indices = torch.where(valid_mask, token_offsets, -1).amax(dim=1)
    safe_indices = last_valid_indices.clamp_min(0).to(torch.int64).unsqueeze(1)
    last_valid_tokens = sampled.gather(1, safe_indices).squeeze(1).to(torch.int32)

    _copy_to_device(
        next_token_ids[:num_reqs],
        torch.where(
            discarded,
            backups,
            torch.where(valid_counts == 0, backups, last_valid_tokens),
        ),
    )
    _copy_to_device(
        valid_sampled_tokens_count[:num_reqs],
        torch.where(discarded, torch.zeros_like(valid_counts), valid_counts),
    )


def _prepare_inputs_padded(
    cu_num_draft_tokens: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    num_rejected_tokens: torch.Tensor,
    num_reqs: int,
) -> None:
    cumulative = cu_num_draft_tokens[:num_reqs].cpu().to(torch.int32)
    previous = torch.zeros_like(cumulative)
    previous[1:] = cumulative[:-1]
    draft_counts = cumulative - previous
    valid_counts = valid_sampled_tokens_count[:num_reqs].cpu().to(torch.int32)
    rejected = torch.where(
        draft_counts > 0,
        draft_counts + 1 - valid_counts,
        torch.zeros_like(draft_counts),
    )
    query_starts = query_start_loc[: num_reqs + 1].cpu().to(torch.int32)
    indices = query_starts[1:] - 1 - rejected

    _copy_to_device(token_indices_to_sample[:num_reqs], indices)
    _copy_to_device(num_rejected_tokens[:num_reqs], rejected)


def _copy_and_expand_eagle_inputs(
    target_token_ids_ptr: torch.Tensor,
    target_positions_ptr: torch.Tensor,
    next_token_ids_ptr: torch.Tensor,
    out_input_ids_ptr: torch.Tensor,
    out_positions_ptr: torch.Tensor,
    out_is_rejected_token_mask_ptr: torch.Tensor,
    out_is_masked_token_mask_ptr: torch.Tensor,
    out_new_token_indices_ptr: torch.Tensor,
    out_hidden_state_mapping_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
    query_end_loc_ptr: torch.Tensor,
    padding_token_id: int,
    parallel_drafting_token_id: int,
    total_input_tokens: int,
    num_padding_slots_per_request: int,
    shift_input_ids: bool,
    BLOCK_SIZE_TOKENS: int,
) -> None:
    del BLOCK_SIZE_TOKENS

    target_tokens_cpu = target_token_ids_ptr[:total_input_tokens].cpu()
    target_positions_cpu = target_positions_ptr.reshape(-1)[:total_input_tokens].cpu()
    next_tokens_cpu = next_token_ids_ptr.cpu()
    query_starts_cpu = query_start_loc_ptr.cpu()
    query_ends_cpu = query_end_loc_ptr.cpu()
    num_reqs = query_ends_cpu.shape[0]

    output_count = total_input_tokens + num_padding_slots_per_request * num_reqs
    if shift_input_ids:
        output_count -= num_reqs
    input_ids_cpu = torch.empty(output_count, dtype=out_input_ids_ptr.dtype)
    positions_cpu = torch.empty(output_count, dtype=out_positions_ptr.dtype)
    rejected_cpu = torch.zeros(output_count, dtype=torch.bool)
    masked_cpu = torch.zeros(output_count, dtype=torch.bool)
    new_indices_cpu = torch.empty(
        num_padding_slots_per_request * num_reqs,
        dtype=out_new_token_indices_ptr.dtype,
    )
    hidden_mapping_cpu = torch.empty(
        total_input_tokens,
        dtype=out_hidden_state_mapping_ptr.dtype,
    )

    for request_index in range(num_reqs):
        query_start = int(query_starts_cpu[request_index])
        next_query_start = int(query_starts_cpu[request_index + 1])
        query_end = int(query_ends_cpu[request_index])
        if shift_input_ids:
            num_valid_tokens = query_end - query_start
            input_offset = 1
            output_start = query_start + request_index * (
                num_padding_slots_per_request - 1
            )
        else:
            num_valid_tokens = query_end - query_start + 1
            input_offset = 0
            output_start = query_start + request_index * num_padding_slots_per_request

        num_rejected = next_query_start - query_end - 1
        total_output_tokens = (
            num_valid_tokens + num_padding_slots_per_request + num_rejected
        )
        start_position = target_positions_cpu[query_start]

        for local_index in range(total_output_tokens):
            output_index = output_start + local_index
            is_bonus = local_index == num_valid_tokens
            is_parallel = (
                num_valid_tokens
                < local_index
                < (num_valid_tokens + num_padding_slots_per_request)
            )
            is_rejected = local_index >= (
                num_valid_tokens + num_padding_slots_per_request
            )

            if local_index < num_valid_tokens:
                input_index = query_start + input_offset + local_index
                token_id = target_tokens_cpu[input_index]
            elif is_bonus:
                token_id = next_tokens_cpu[request_index]
            elif is_parallel:
                token_id = parallel_drafting_token_id
            else:
                token_id = padding_token_id

            input_ids_cpu[output_index] = token_id
            positions_cpu[output_index] = (
                0 if is_rejected else start_position + local_index
            )
            rejected_cpu[output_index] = is_rejected
            masked_cpu[output_index] = is_parallel

            if (
                num_valid_tokens
                <= local_index
                < (num_valid_tokens + num_padding_slots_per_request)
            ):
                new_token_index = (
                    request_index * num_padding_slots_per_request
                    + local_index
                    - num_valid_tokens
                )
                new_indices_cpu[new_token_index] = output_index

        if shift_input_ids:
            num_input_tokens_for_request = next_query_start - query_start
            for local_index in range(num_input_tokens_for_request):
                hidden_mapping_cpu[query_start + local_index] = (
                    output_start + local_index
                )

    _copy_to_device(out_input_ids_ptr[:output_count], input_ids_cpu)
    _copy_to_device(out_positions_ptr.reshape(-1)[:output_count], positions_cpu)
    _copy_to_device(out_is_rejected_token_mask_ptr[:output_count], rejected_cpu)
    _copy_to_device(out_is_masked_token_mask_ptr[:output_count], masked_cpu)
    _copy_to_device(
        out_new_token_indices_ptr[: new_indices_cpu.numel()], new_indices_cpu
    )
    if shift_input_ids:
        _copy_to_device(
            out_hidden_state_mapping_ptr[:total_input_tokens], hidden_mapping_cpu
        )


def eagle_step_update_slot_mapping_and_metadata(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
    out_clamped_positions: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    input_batch_size: int | None = None,
) -> None:
    positions_cpu = positions_1d.cpu().to(torch.int32)
    block_table_cpu = block_table_tensor.cpu().to(torch.int32)
    seq_lens_cpu = seq_lens.cpu().to(torch.int32)
    batch_size = positions_cpu.shape[0]
    if input_batch_size is None:
        input_batch_size = batch_size

    new_positions = positions_cpu + 1
    exceeds_max = new_positions >= max_model_len
    clamped_positions = torch.where(
        exceeds_max, torch.zeros_like(new_positions), new_positions
    )
    block_numbers = (clamped_positions // block_size).clamp_max(
        block_table_cpu.shape[1] - 1
    )
    request_indices = torch.arange(batch_size, dtype=torch.int64)
    block_ids = block_table_cpu[request_indices, block_numbers.to(torch.int64)]
    slot_mapping = block_ids * block_size + clamped_positions % block_size
    slot_mapping = torch.where(
        exceeds_max, torch.full_like(slot_mapping, PADDING_SLOT_ID), slot_mapping
    )
    new_seq_lens = torch.where(
        exceeds_max, torch.ones_like(seq_lens_cpu), seq_lens_cpu + 1
    )
    new_seq_lens.clamp_max_(max_model_len)

    padded_slot_mapping = torch.full(
        (input_batch_size,), PADDING_SLOT_ID, dtype=out_slot_mapping.dtype
    )
    padded_slot_mapping[:batch_size] = slot_mapping.to(out_slot_mapping.dtype)
    _copy_to_device(out_clamped_positions[:batch_size], clamped_positions)
    _copy_to_device(out_slot_mapping[:input_batch_size], padded_slot_mapping)
    _copy_to_device(seq_lens[:batch_size], new_seq_lens)


eagle_prepare_next_token_padded_kernel = _GridLaunchable(_prepare_next_token_ids_padded)
eagle_prepare_inputs_padded_kernel = _GridLaunchable(_prepare_inputs_padded)
copy_and_expand_eagle_inputs_kernel = _GridLaunchable(_copy_and_expand_eagle_inputs)

_shim_installed = False


def install() -> None:
    """Install QAIC-compatible speculative-input helpers."""
    global _shim_installed
    if _shim_installed:
        return

    import vllm.v1.spec_decode.llm_base_proposer as llm_base_proposer
    import vllm.v1.spec_decode.utils as spec_decode_utils

    replacements = {
        "eagle_prepare_next_token_padded_kernel": (
            eagle_prepare_next_token_padded_kernel
        ),
        "eagle_prepare_inputs_padded_kernel": eagle_prepare_inputs_padded_kernel,
        "copy_and_expand_eagle_inputs_kernel": copy_and_expand_eagle_inputs_kernel,
        "eagle_step_update_slot_mapping_and_metadata": (
            eagle_step_update_slot_mapping_and_metadata
        ),
    }
    for name, replacement in replacements.items():
        setattr(llm_base_proposer, name, replacement)
        setattr(spec_decode_utils, name, replacement)

    _shim_installed = True
    logger.info(
        "vllm_qaic: QAIC PYT speculative-input compatibility helpers installed."
    )

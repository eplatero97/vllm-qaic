# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import torch

from vllm_qaic.v1.spec_decode.input_shim import (
    copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata,
)


def test_prepare_next_token_ids_padded_handles_acceptance_and_discard():
    sampled_token_ids = torch.tensor(
        [[10, 11, -1, -1], [20, 21, 22, -1], [30, -1, -1, -1]],
        dtype=torch.int32,
    )
    discard_request_mask = torch.tensor([False, True, False])
    backup_next_token_ids = torch.tensor([100, 101, 102], dtype=torch.int32)
    next_token_ids = torch.empty(3, dtype=torch.int32)
    valid_counts = torch.empty(3, dtype=torch.int32)

    eagle_prepare_next_token_padded_kernel[(3,)](
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_counts,
        32000,
        4,
        3,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=4,
    )

    assert next_token_ids.tolist() == [11, 101, 30]
    assert valid_counts.tolist() == [2, 0, 1]


def test_prepare_inputs_padded_accounts_for_rejected_tokens():
    cumulative_draft_tokens = torch.tensor([3, 5], dtype=torch.int32)
    valid_counts = torch.tensor([2, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32)
    token_indices = torch.empty(2, dtype=torch.int32)
    rejected_counts = torch.empty(2, dtype=torch.int32)

    eagle_prepare_inputs_padded_kernel[(2,)](
        cumulative_draft_tokens,
        valid_counts,
        query_start_loc,
        token_indices,
        rejected_counts,
        2,
    )

    assert rejected_counts.tolist() == [2, 2]
    assert token_indices.tolist() == [1, 4]


def test_copy_and_expand_inputs_for_draft_model():
    target_token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    target_positions = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    next_token_ids = torch.tensor([10, 20], dtype=torch.int32)
    output_token_ids = torch.empty(6, dtype=torch.int32)
    output_positions = torch.empty(6, dtype=torch.int64)
    rejected_mask = torch.empty(6, dtype=torch.bool)
    masked_token_mask = torch.empty(6, dtype=torch.bool)
    new_token_indices = torch.empty(2, dtype=torch.int32)
    hidden_state_mapping = torch.empty(4, dtype=torch.int32)

    copy_and_expand_eagle_inputs_kernel[(2, 1)](
        target_token_ids_ptr=target_token_ids,
        target_positions_ptr=target_positions,
        next_token_ids_ptr=next_token_ids,
        out_input_ids_ptr=output_token_ids,
        out_positions_ptr=output_positions,
        out_is_rejected_token_mask_ptr=rejected_mask,
        out_is_masked_token_mask_ptr=masked_token_mask,
        out_new_token_indices_ptr=new_token_indices,
        out_hidden_state_mapping_ptr=hidden_state_mapping,
        query_start_loc_ptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        query_end_loc_ptr=torch.tensor([1, 3], dtype=torch.int32),
        padding_token_id=0,
        parallel_drafting_token_id=0,
        total_input_tokens=4,
        num_padding_slots_per_request=1,
        shift_input_ids=False,
        BLOCK_SIZE_TOKENS=4,
    )

    assert output_token_ids.tolist() == [1, 2, 10, 3, 4, 20]
    assert output_positions.tolist() == [0, 1, 2, 0, 1, 2]
    assert rejected_mask.tolist() == [False] * 6
    assert masked_token_mask.tolist() == [False] * 6
    assert new_token_indices.tolist() == [2, 5]


def test_step_update_slot_mapping_and_metadata():
    positions = torch.tensor([3, 7], dtype=torch.int64)
    block_table = torch.tensor([[5, 6], [8, 9]], dtype=torch.int32)
    seq_lens = torch.tensor([4, 8], dtype=torch.int32)
    output_positions = torch.empty(2, dtype=torch.int64)
    slot_mapping = torch.empty(3, dtype=torch.int64)

    eagle_step_update_slot_mapping_and_metadata(
        positions,
        block_table,
        seq_lens,
        block_size=4,
        max_model_len=8,
        out_clamped_positions=output_positions,
        out_slot_mapping=slot_mapping,
        input_batch_size=3,
    )

    assert output_positions.tolist() == [4, 0]
    assert slot_mapping.tolist() == [24, -1, -1]
    assert seq_lens.tolist() == [5, 1]

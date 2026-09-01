# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import argparse
import json
import os
from pathlib import Path

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "mtp"))
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tensor_parallel_size = int(
        os.environ.get("QAIC_EAGER_MTP_TENSOR_PARALLEL_SIZE", "1")
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32, seed=42)
    kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        max_num_seqs=1,
        max_model_len=64,
        long_prefill_token_threshold=128,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.99,
        disable_log_stats=False,
    )
    if args.mode == "mtp":
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": 1,
        }

    llm = LLM(**kwargs)
    try:
        outputs = llm.generate(
            [
                "The capital of France is",
                "A short story begins with a quiet morning when",
            ],
            sampling_params,
        )
        spec_decode_metrics = {}
        for metric in llm.get_metrics():
            if metric.name in {
                "vllm:spec_decode_num_draft_tokens",
                "vllm:spec_decode_num_accepted_tokens",
            }:
                spec_decode_metrics[metric.name] = (
                    spec_decode_metrics.get(metric.name, 0) + metric.value
                )
        result = {
            "mode": args.mode,
            "model": args.model,
            "spec_decode_metrics": spec_decode_metrics,
            "token_ids": [list(output.outputs[0].token_ids) for output in outputs],
        }
        args.output_path.write_text(json.dumps(result))
    finally:
        del llm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

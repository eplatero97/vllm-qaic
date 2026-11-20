# Task: is any hybrid or linear-attention model supported in PYT eager mode by default?

## Objective
Find out whether **any** model that is not a plain full-attention model already runs in QAIC PYT (eager /
torch.compile) mode with little or no code change. Then determine whether ngram/suffix speculative decoding (SpD)
works on top of it.

This decides the fate of the raise added in PR #106 at `vllm_qaic/platform.py:452-464`. That raise rejects PYT +
SpD when `model_config.is_hybrid or model_config.is_attention_free`. Reviewer `quic-agokhale` asked for it to be
removed if upstream support is inherited (thread 3, comment id 4637027352). We kept it because the one model we
tried failed **without** SpD.

"Hybrid / linear attention" covers two different families. Investigate both, and report them separately:

| Family | Examples | KV/state mechanism in vLLM 0.30.0 | QAIC concern |
|---|---|---|---|
| A. Recurrent / linear-attention hybrids | Mamba2 hybrids (granite-4.0-h, Falcon-H1, Zamba2, Nemotron-H), Jamba (Mamba1), LFM2 (short conv), GDN (Qwen3-Next / Qwen3.5) | `MambaSpec` state slots, mamba/GDN/conv Triton kernels | Triton kernels must compile and run on `qcom_hexagon_backend` |
| B. Hybrid *attention* (sliding window + full) | Gemma2/Gemma3 text, gpt-oss, Mistral SWA, Phi-3-small | `SlidingWindowSpec` + `FullAttentionSpec` KV groups | the native paged-attention path rejects sliding window (`qaic_attn.py`: `sliding_window[1] not in (-1, 0)` → SDPA fallback); does the SDPA fallback honour the window? |
| C. Pure attention-free (for completeness) | `state-spaces/mamba-130m-hf`, Mamba2-130m | `MambaSpec` only | upstream itself skips the SpD accepted-token update for these, so non-SpD support only |

## What is already known (from PR #106, see `docs/qaic/pr106_followup_review_responses.md`, thread 3)
- **Upstream SpD on hybrids does not roll state back.** It keeps one state slot per speculative position
  (`MambaSpec.num_speculative_blocks`, `vllm/model_executor/layers/mamba/abstract.py:~81-85`;
  `state_indices_tensor_d` of shape (bs, 1+k), `vllm/v1/attention/backends/mamba_attn.py:~162-166`). The runner
  records `num_accepted_tokens` (`vllm/v1/worker/gpu_model_runner.py:~1587-1645, 2114-2148, 2528-2545`; gated on
  `is_hybrid`). Triton kernels then pick the right slot: `causal_conv1d_update` (with `IS_SPEC_DECODING`),
  `selective_state_update` and `fused_sigmoid_gating`.
- `QaicModelRunnerPyt(GPUModelRunner)` (`vllm_qaic/worker/model_runner.py`) overrides **none** of this. Only the AOT
  runner overrides `get_kv_cache_spec` (`FullAttentionSpec` only). Mamba/GDN layers choose their own backend via
  `MambaBase.get_attn_backend`.
- **Measured:** `ibm-granite/granite-4.0-h-350m` (Mamba2 hybrid) fails in PYT **without SpD**, inside the Triton
  `_chunk_state_fwd_kernel` (Mamba2 SSD chunked prefill). The exact error text and whether the failure is a
  compile error, a runtime error or a numerical error were **not** characterized. Look in the findings log
  `docs/qaic/pyt_ngram_suffix_eager_findings.md` (section "W2: hybrid models") and in `runs/` / `experiments/` for
  the original log before re-running.
- **Untested:** GDN, Mamba1, LFM2, Zamba2, Falcon-H1, and the whole of family B.
- `docs/docs/user_guide/models/supported_models_eager.md` lists what is validated in eager mode. No hybrid model is
  listed.
- `platform.py:~48-49` mentions Qwen3.5, but that model is multimodal and hits the multimodal assert.
- Earlier, the HF cache (`HF_HOME=/local/mnt/workspace/eplatero/hf`) held only TinyLlama. Expect to download models.
  Check free disk space first; `/tmp` is RAM tmpfs.

## Questions to answer
1. For each family, which (if any) small model runs end to end in PYT on one QID with fp16 and greedy decoding,
   with **zero** code changes? Use 24-64 tokens, and compare against an HF CPU/fp32 reference for token agreement.
2. For each failure, classify it:
   - a Triton compile error on `qcom_hexagon_backend`;
   - a runtime/device error;
   - a missing QAIC op that falls back to CPU (`QAIC_DEBUG=1`);
   - a QAIC-platform gate or assert (an explicit `vllm_qaic` rejection);
   - wrong output.

   Give the exact kernel or op name and the file:line.
3. Is each failure **"little change"** or **"real work"**? Examples of little change:
   - a missing `pre_register` route;
   - a dtype cast;
   - a kernel already supported with a different `BLOCK_SIZE` or `num_warps`;
   - a known Hexagon-Triton limitation that has a workaround already used elsewhere in `vllm_qaic`.

   Examples of real work: a new custom kernel, or an unsupported Triton feature (e.g. `tl.dot` with an unsupported
   shape, atomics, `tl.associative_scan`). Estimate the size of each.
4. For family B: does the sliding-window layer take the SDPA fallback, and does that fallback **apply the window**?
   If it silently attends to the full context, outputs will diverge once the context exceeds the window. Test with
   a prompt plus generation longer than the window, or use a model or config with a small window.
5. For any model that passes question 1, run ngram SpD with the raise bypassed. Use an env-guarded local patch that
   is **not committed**. Check greedy equivalence against no-SpD, and check `0 < accepted < drafted` using
   rejection-forcing prompts (reuse `_PROMPTS` from `tests/e2e/spec_decode/test_qaic_ngram.py`).
6. Based on the results, which raise is correct? Choose one:
   - keep it;
   - narrow it to `is_attention_free` plus the specific failing families;
   - remove it.

## Suggested model shortlist (smallest first; confirm the HF ids exist and fit on 1 QID in fp16)
- A, Mamba2 hybrid: `ibm-granite/granite-4.0-h-350m` (the known failure, rerun to capture the exact error),
  `tiiuae/Falcon-H1-0.5B-Instruct`, `Zyphra/Zamba2-1.2B`.
- A, short-conv hybrid: `LiquidAI/LFM2-350M`. It uses `ShortConv` and `causal_conv1d`, without the SSD prefill
  kernel, so it could pass even where Mamba2 fails.
- A, Mamba1 hybrid: `ai21labs/Jamba-tiny-dev` (or another tiny Jamba).
- A, GDN: look for a small text-only Qwen3-Next / GDN checkpoint. If only large ones exist, test the GDN Triton
  kernels directly (e.g. `fla` chunk/fused-recurrent ops in `vllm/model_executor/layers/fla/`) with a unit-style
  harness on QAIC.
- B: `google/gemma-3-270m` or `google/gemma-2-2b-it` (sliding window + global), and a Mistral-style SWA model if a
  small one exists.
- C: `state-spaces/mamba-130m-hf` (non-SpD only).

When a model cannot be loaded at all for a reason unrelated to QAIC (an architecture missing in vllm 0.30.0, or a
gated repo), note it and move on.

## Method
1. **Preflight.** Follow CLAUDE.md "PYT (Eager) Mode":
   - use `.venv_eager`;
   - confirm the Qualcomm Triton 3.7.0 with `qcom_hexagon_backend`;
   - confirm the editable install, using the worktree
     `/prj/crd/austin/validation/scratch/users/eplatero/vllm/vllm-qaic-pyt-ngram-suffix` on branch
     `pyt-ngram-suffix-spd` at `14ee2a9`, or create a new task worktree from it;
   - source `/tmp/pyt_env.sh`, re-creating it from CLAUDE.md if it is missing;
   - set `QAIC_NUM_CORES=16`, `QAIC_NUM_THREADS=4` and `QAIC_DEBUG=1`.
2. **Devices.** Run `qaic-util | grep -i "qid\|status\|board serial"`. Use one free QID per model through
   `QAIC_VISIBLE_DEVICES`, and run models in parallel on disjoint QIDs. Use `VLLM_WORKER_MULTIPROC_METHOD=spawn`
   for any script that builds more than one LLM per process. Under fork, the forked EngineCore inherits the
   frontend's `/dev/accel` fd, the NSPs stay reserved, and the next LLM fails with "0 free nsps".
3. **Kernel-level triage first** (it is cheap). For each family, find the Triton kernels on the model's forward
   path in vllm 0.30.0:
   - `vllm/model_executor/layers/mamba/ops/` (ssd_*, causal_conv1d, selective_state_update);
   - `vllm/model_executor/layers/fla/`;
   - short-conv ops.

   Write a small harness under `experiments/pyt_hybrid_kernels/` that calls each kernel on QAIC tensors with tiny
   shapes and compares it against the PyTorch/CPU reference implementation, using upstream's reference functions
   where they exist. This separates "the kernel does not compile on Hexagon" from "the integration is broken", and
   it gives a per-kernel compatibility table. A read-only subagent can map the kernel inventory in parallel.
4. **Model-level runs.** Use one script per model under `experiments/pyt_hybrid_models/`:
   - (a) PYT, no SpD, greedy, compared against HF (CPU, fp32 or bf16) for token agreement;
   - (b) only if (a) passes, ngram (then suffix) SpD with the raise bypassed;
   - (c) greedy comparison of (a) against (b);
   - (d) the acceptance counters (`vllm:spec_decode_num_draft_tokens` / `..._accepted_tokens` deltas).

   Run them in the background and save logs under `runs/pyt_hybrid/`.
5. **Family B window check.** Run a generation longer than the sliding window, and compare against HF. Separately,
   confirm from the code that the SDPA fallback path in `qaic_attn.py` applies the window.
6. **Methodology review.** Use the `methodology-review` skill, preferably as an independent subagent, on the kernel
   harness and on each pass/fail claim. Possible flaws to look for:
   - comparing only a few tokens;
   - a reference that shares the same kernel;
   - context that is too short to exceed the window or the SSD chunk size (e.g. `chunk_size` 256 for Mamba2:
     prompts must be longer than one chunk to exercise the multi-chunk path);
   - a pass caused by a CPU fallback that hides a missing QAIC kernel.

## Decision rules
- Do not remove or narrow the raise for a family unless a model from that family passes (a), and also passes (b)
  through (d) with real rejections (`0 < accepted < drafted`) and token-exact greedy equality.
- If a family passes (a) but diverges under SpD, keep the raise for that family and record the divergence.
- If a family fails (a) only because of a small, contained fix (for example, one kernel needs a tweak), describe the
  fix with a size estimate and **ask the user** before implementing it. Do not implement kernel changes
  unprompted.
- Family C stays excluded under SpD whatever the results, because upstream excludes it.

## Deliverables
- A findings log: `docs/qaic/pyt_hybrid_linear_attention_support.md` (flat, untracked, kept up to date as you go).
  Follow the CLAUDE.md structure: Summary/Status, Objective, Test config, per-family results, root causes marked
  CONFIRMED vs hypothesized, Follow-up. Explain the recurrent-state and sliding-window mechanisms from first
  principles for readers who were not present. Include:
  - a per-kernel table: kernel, family, compiles on Hexagon, matches reference, CPU fallbacks seen;
  - a per-model table: model, family, (a) result, (b)-(d) results, failure class, fix size.
- A recommendation for the PR #106 raise (keep / narrow / remove), with evidence, plus the updated reply text for
  thread 3.
- If a narrowing is justified: the proposed `platform.py` diff and a dual-mode e2e test case. Do not commit without
  asking.
- If any Family A or B model works without SpD, propose a `supported_models_eager.md` update. Ask first.

## Rules
- Do not push, open PRs, or post GitHub replies without asking the user.
- Commit only on request: `git commit -s` with the `Co-Authored-By` trailer from the session reminder. Do not stage
  new `*.md` files, `experiments/` or `runs/`.
- Only ruff (`ruff check` / `ruff format --check`) must pass on any code diff.
- Always run pytest with `-s`. Run e2e and model runs in the background. Long downloads and compiles: mind the 2 h
  background-task cap (detach if needed).
- Give a backward-compatibility statement for any proposed change to the raise. It affects every PYT SpD user, and
  AOT is unaffected (the check sits in the PYT branch).
- Send a completion email with `~/.claude/hooks/notify_email.sh` when finished.

## Subagent decomposition (suggested)
- Main path: preflight, model downloads, kernel harness, then integration and the decision.
- Parallel, read-only research:
  - (1) inventory the Triton kernels on the forward path of each family in vllm 0.30.0, with the Triton features
    each kernel uses (`tl.dot` shapes, atomics, scans);
  - (2) does `qaic_attn.py`'s SDPA fallback apply the sliding window, and how does `QaicModelRunnerPyt` handle
    hybrid KV-cache groups;
  - (3) any known `qcom_hexagon_backend` limitations listed in `docs/qaic/build_hexagon_triton_backend.md` or the
    pytorch repo's Triton tests.
- Parallel hardware: one background run per model on disjoint QIDs.
- Methodology-review subagent on the final evidence.

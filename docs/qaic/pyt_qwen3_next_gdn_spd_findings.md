# Qwen3-Next (Gated DeltaNet) in PYT eager on QAIC, and native SpD state caching with ngram/suffix

## Summary / Status
- **Status:** COMPLETE (2026-11-20). Methodology review: see the end of the log.
- **Q1:** does a Qwen3-Next model run end to end in PYT eager on one QID (fp16, greedy) with zero code changes?
  **NO with zero changes. YES (deterministic, 6/8 prompts token-exact vs HF fp32 over 64 tokens, the other 2
  diverge at measured fp16 near-ties) with these experiment-side shims (runs 1-18):**
  1. `torch.accelerator.empty_cache` shim (PYT runner calls it; not implemented for QAIC).
  2. `QAicRMSNormGated.hidden_size` → `weight.shape[-1]`.
  3. `QaicPlatform.support_hybrid_kv_cache()` → True.
  4. Kernel-addressable GDN state pool: cap the KV budget (e.g. 1 GiB). A ~15 GB pool view faults the kernels.
     Warmup is also skipped; it probably hit the same fault.
  5. **vllm_qaic bug, also on `main`:** the `QAicGemmaRMSNorm` residual path passes the raw zero-centred
     weight to the Hexagon kernel; it must pass `1 + weight`. This also affects Gemma models.
  6. **vllm_qaic bug, also on `main`:** `QAicApplyRotaryEmb` / `qaic.rotary_embedding` gives wrong results for
     a strided multi-head input (`query[..., :rotary_dim]`): rel err 1.06 at H=2, 1.80 at H=4. `.contiguous()`
     fixes it. This affects every partial-rotary model.
  7. **Hexagon Triton backend:** out-of-range lanes of masked / boundary-checked `tl.load` hold uninitialized
     memory. This made output nondeterministic. A global zero-fill shim (`padding_option="zero"`, `other=0`)
     fixes it.
  8. Not a QAIC issue: this checkpoint has `tie_word_embeddings=true` and no `lm_head.weight`, but upstream
     `Qwen3NextForCausalLM` never ties, so lm_head must be tied manually.
- **Q2:** does upstream's native per-position recurrent-state caching give greedy output identical to non-SpD
  for ngram and suffix? **NO, as shipped.** Two independent causes (runs 19-22):
  - **H1, upstream gap (code-confirmed; MEASURED on QAIC at mns=1):** a request with zero drafts (no
    ngram/suffix match) takes the GDN non-spec path, which reads state slot `block_table[:,0]` and ignores
    `num_accepted_tokens`. After a step that accepted a>1 tokens, the valid state is in slot a-1, so the read
    is stale (SSM and conv state). It affects proposers that can emit zero drafts. By code reading, it would
    also affect GPU (source audit only, not executed) in `mamba_cache_mode` "none" and, unless a block boundary is crossed, "align". This was
    not run on GPU.
  - **H2, `causal_conv1d_fn` conv-state bug observed on QAIC (MEASURED by microtest):** in batches that mix
    spec and non-spec rows, upstream routes 1-token decodes through `causal_conv1d_fn` with an initial state.
    On QAIC that kernel writes zeros into the conv-state window for sequences shorter than width-1 (3). It
    only matters for mns>1. **Not yet shown to be QAIC-specific:** it was compared only against a torch
    reference, not the same Triton kernel on another backend.
  - H1 and H2 are not causally separated end to end. The padded control removes both at once.
  - **The native slot mechanism itself works on QAIC:** with every proposal padded to k drafts (no
    non-spec rows), output equals baseline on 5-7 of 8 prompts. Each remaining divergence is at a measured
    fp16 near-tie (HF top-2 margin 0.004-0.017).
  - Fix directions (follow-up, not done): keep zero-draft decode rows on the multi-slot spec path (as the
    Mamba2 builder does) or copy slot a-1 → slot 0; pad drafts; fix the Hexagon Triton `causal_conv1d_fn`
    short-sequence branch.

## How each conclusion was reached (evidence trail)
This section traces each conclusion from the first hypothesis through each test to its current status. The
run numbers match the Phase log below. The exact commands, QIDs and output files are in
`experiments/pyt_qwen3next_spd/README.md` on branch `exp/pyt-qwen3next-gdn-spd`.
Status labels: **CONFIRMED** = measured on device, with a control. **MEASURED** = observed, without an
isolating control. **CODE-READ** = established by reading source only.

```
Q1 path:  run1 -> fix1 -> run2 -> fix2 -> run3 (MoE fail, switch model) -> run4 -> fix3 -> run5/5loc -> run6
          -> run7 (conv fault) -> run8 (KV cap: fault gone, garbage) -> run10 (Gemma norm) -> run11/12 (RoPE)
          -> run13 (lm_head tie) -> run14 (nondeterministic) -> runs17-18 (tl.load zero-fill: deterministic) = Q1 YES
Q2 path:  plan (H1 code-read) -> Phase1 hazard probe (H1 kernel-level) -> runs 15/16 (discarded) -> runs 19-21
          (deterministic baseline; SpD diverges; H1 timing at mns=1; early divergence at mns=8 -> H2)
          -> H2 microtest -> run 22 padded control -> methodology review = Q2 NO
```

### C1. Qwen3-Next does not run with zero changes (CONFIRMED)
- **Starting point:** the plan assumed only GDN kernels might fail, since Qwen3-Next uses a kernel set
  disjoint from granite's Mamba2 `_chunk_state_fwd_kernel`.
- **Tests:** runs 1-9 each stopped at the next failure:
  - host `empty_cache` assert (run 1);
  - `QAicRMSNormGated.hidden_size` (run 2);
  - MoE HMX error 500 on tiny-random (run 3, out of scope, so the 70M model was used from then on);
  - KV-spec promotion (run 4);
  - warmup error 500 (run 5);
  - conv1d device fault (runs 5-loc, 6, 7);
  - garbage output (runs 8-9).
- **History:** the first failures were all host-side or in vllm_qaic, not GDN kernels. That contradicted the
  starting assumption.

### C2. The ~15 GB GDN state pool faults the kernels; a 1 GiB KV cap fixes it (MEASURED)
- **Evidence:**
  - run 7 trace: the fault is inside `causal_conv1d_fn`, on a ~15 GB `conv_states` view;
  - run 8: the same command with `QWEN3NEXT_KV_BYTES=1GiB` gives no fault.
- **Limit:** the exact addressable-size threshold was not bisected. The warmup is skipped on the assumption
  that it hit the same fault; that was not proven.

### C3. Two vllm_qaic bugs, also on `main`: GemmaRMSNorm residual weight, and strided RoPE (CONFIRMED)
- **How found:** output was garbage after run 8, so run 10 diffed vLLM against HF fp32 per sublayer. The
  first divergence was the GemmaRMSNorm residual path (rel 0.5). The shim passes `1 + weight`, and run 11
  shows layers 0-2 then match.
- The next divergence was layer-3 attention. `attn_probe.py` narrowed it to RoPE, and `rope_micro.py`
  reproduced it standalone: `query[..., :rotary_dim]` gives rel 1.06 at H=2 and 1.80 at H=4, and
  `.contiguous()` fixes it.
- **Control:** run 13, with both shims on, shows all layers within 4.9e-3 rel of HF.

### C4. Untied lm_head (CONFIRMED, not a QAIC issue)
Run 13's layer diff showed every layer matching while the logits did not. The checkpoint sets
`tie_word_embeddings=true` and has no `lm_head.weight`, but upstream `Qwen3NextForCausalLM` never ties them.
The tie shim fixes it (run 14).

### C5. Nondeterminism comes from uninitialized out-of-bounds `tl.load` lanes on Hexagon Triton (CONFIRMED)
- **History:**
  - Phase 1 found `chunk_gated_delta_rule` gives NaN unless the length is a multiple of 64. The pad ablation
    showed that zero-padding the k-load alone fixes it, giving `QWEN3NEXT_ZERO_PAD`.
  - Run 14 was still nondeterministic across two baseline runs. Hypothesis: the same uninitialized-lane
    behaviour in other kernels.
  - Runs 17-18 applied a global zero-fill (`padding_option="zero"`, `other=0`) with a fresh Triton cache.
    The result was 64/64 deterministic.
- **Control:** baseline vs baseline2 within one run, at mns=8 (run 17-18) and mns=1 (run 19).
- **Consequence:** runs 15-16, the first SpD runs, came before this fix. They were discarded because their
  baseline was not deterministic.

### C6. Q1 = YES with the 8 shims (CONFIRMED, limited to one 70M model)
- Run 17-18 (mns=8): 6/8 prompts are token-exact vs HF fp32 over 64 tokens.
- Run 19 (mns=1): 7/8 exact.
- `margin_check.py`: every divergence is where HF's top-2 margin is tiny (0.0172, 0.0039), and vLLM picked
  HF's rank-1 token.

### C7. Q2 = NO as shipped (CONFIRMED for this model and config)
- Runs 20 (mns=1) and 21 (mns=8): SpD vs the deterministic baseline of the same engine.
  - Identical prompts: ngram 1/8 and suffix 2/8 at mns=1; 0/8 for both at mns=8.
  - Drafts / accepted: 671/130 and 499/228 at mns=1, so drafts were being accepted, not ignored.
- **History:** this outcome was case (b) or (c) in the plan. Separating the two was the job of H1 and H2.

### C8. H1: zero-draft requests read stale state slot 0 (upstream gap; CODE-READ + kernel-level CONFIRMED + E2E MEASURED)
1. **Plan (code-read, vLLM 0.23):** a request with no drafts gets `num_decode_draft_tokens=-1`. It takes the
   non-spec path, which reads `block_table[:,0]` with no `num_accepted_tokens`. After a step that accepted
   a>1 tokens, the valid state is in slot a-1.
2. **Phase 1 `hazard` stage (kernel level):** the non-spec output matches the stale-state reference (err
   2.9e-6), not the correct one (6.9e-3).
3. **Run 20 + `hazard_analyze.py` (E2E, mns=1, no mixed batches):**
   - in 13/14 diverging requests, divergence starts at the first hazard or 1-7 tokens after it;
   - no request diverges before its first hazard;
   - the only hazard-free divergence is a measured near-tie.
   - The first analyzer (inside the harness) reported 0 hazards because it skipped `draft=None` records.
     The offline `hazard_analyze.py` fixed that and clusters the per-KV-group copies of each record.
4. **Read-only code agent (vLLM 0.30.0):** confirmed the gap in "none" mode, and in "align" mode unless the
   step crosses a block boundary. It also found that the Mamba2 builder avoids the gap, and that no upstream
   code or test covers it.
5. **Run 22 at mns=1 (padded control):** identical prompts went from 1-2/8 to 5-6/8; the rest are near-ties.
- **Limits:**
  - No targeted H1-only fix (copy slot a-1 → 0) was run.
  - The GPU claim is source audit only.

### C9. H2: `causal_conv1d_fn` zeroes the conv-state window for sequences shorter than 3 tokens, on QAIC (kernel CONFIRMED; QAIC-specificity UNVERIFIED)
1. **Run 21 (mns=8):** divergence starts at output index 2-6, before any H1 hazard in 14/16 requests.
   H1 cannot explain this.
2. **Code read:** in mixed batches, upstream reclassifies 1-token decodes as prefills, so they run
   `causal_conv1d_fn` with an initial state. Probe records show mixed steps from the first decode step.
3. **Microtest:** the conv-state write-back is wrong for L<3 (`[old1, old2, 0]`); the outputs and SSM state
   are correct. The bug is unchanged without the zero-fill shim.
4. **Run 22 at mns=8 (padded, 0 mixed steps):** 6-7/8 identical, from 0/8.
- **Limits:**
  - The microtest compared against a torch reference, not the same Triton kernel on another backend. It
    could be an upstream kernel bug.
  - H1 and H2 were not separated end to end.

### C10. The native per-position slot mechanism is correct on QAIC up to fp16 near-ties (MEASURED, weak)
- **Phase 1:** the `spec` and `chain` stages are OK, covering 2D state indices and `num_accepted_tokens`
  slot selection.
- **Run 22:** every remaining divergence is at an HF top-2 margin of 0.0039-0.0172. These were re-verified
  when the archive was made: p1@17 = 0.0092, p6@27 = 0.0039.
- **Limits:**
  - The padded control only exercises all-spec q=5 batches.
  - The near-tie check uses HF margins, not vLLM's logprobs.
  - The acceptance histogram was not reported.

### C11. Methodology review: accepted with limits
See the end of this log for what the review does and does not support, and for the open controls.

## Objective
granite-4.0-h (Mamba2 hybrid) failed in PYT without SpD inside `_chunk_state_fwd_kernel`. That is a Mamba2-only
prefill kernel (`mamba/ops/ssd_chunk_state.py`). It is **not** SpD-specific, and Qwen3-Next does **not** use it.
Qwen3-Next's Gated DeltaNet (GDN) layers use a separate set of Triton kernels, so the granite result says
nothing about Qwen3-Next. This log answers Q1 and Q2 for Qwen3-Next only.

## Background: how a hybrid model carries state, and how upstream SpD handles it
An attention layer stores one K/V entry per token. Rejecting draft tokens just means ignoring their entries. A
GDN layer instead keeps one fixed-size *recurrent state* per request, which every token overwrites. Rejected
tokens would therefore corrupt it. Upstream avoids rollback by keeping **k+1 state slots** per request:

```
step with k=3 drafts:   tokens  [t0  d1  d2  d3]
state written per token  slot:   0   1   2   3       (spec_state_indices row, shape (k+1,))
sampler accepts a tokens -> num_accepted_tokens = a
next spec step reads its initial state from slot a-1
```

- `MambaSpec.num_speculative_blocks = k` allocates the slots.
- The GPU runner records `num_accepted_tokens`. This is gated only on `model_config.is_hybrid`, not on the SpD
  method.
- These kernel branches run only under SpD:
  - `causal_conv1d_update` with `IS_SPEC_DECODING` (shifts the conv window by `a-1`);
  - `fused_sigmoid_gating_delta_rule_update` with a 2D `ssm_state_indices` (reads slot `a-1`, writes slot t).

**Suspected gap for ngram/suffix (hypothesis, code-read only):**
1. When ngram/suffix finds no match, the request gets zero drafts.
2. It is then absent from `scheduled_spec_decode_tokens`, so the runner sets `num_decode_draft_tokens = -1`.
3. The GDN metadata builder routes it to the **non-spec** path, which reads slot `block_table[:,0]` with
   `num_accepted_tokens=None`.
4. If the previous step accepted more than 1 token, the valid state may be in slot `a-1`, not slot 0.

MTP/EAGLE always propose k tokens and would rarely hit this; ngram/suffix often propose zero.

## Test config
| Item | Value |
|---|---|
| Checkout | `../vllm-qaic-pyt-ngram-suffix` (PR #106 branch `pyt-ngram-suffix-spd`, 14ee2a9), editable |
| venv | its `.venv_eager`: vLLM 0.30.0, transformers 5.23, Qualcomm triton 3.7.0, arctic_inference |
| Env | `/tmp/pyt_env.sh` (Hexagon tool paths, `QAIC_NUM_CORES=16`, `QAIC_NUM_THREADS=4`, `QAIC_DEBUG=1`) |
| Devices | QID 8 (E2E), QID 9 (kernel probe); all QIDs Ready |
| Crash-test model | `tiny-random/qwen3-next-moe`: hidden 8, 4 layers (LLL+F), 16 v-heads/8 k-heads × 128, attn head_dim 32, 32 experts top-10, random weights |
| Agreement model | `theo77186/Qwen3-Next-70M-TinyStories`: hidden 256, 8 layers (LLLF×2), 2 v-heads/1 k-head × 128, conv 4, attn head_dim 256, 16 experts top-2, trained |

## Phase log
(updated as work proceeds)

### 2026-11-20 — Phase 0 (setup): OK
- PYT preflight passed: Qualcomm triton 3.7.0, `qcom_hexagon_backend` present, editable install resolves to the
  worktree, vLLM 0.30.0.
- QIDs 0-19 all Ready.
- Both models downloaded.
- HF CPU fp32 reference for the 70M model runs, using transformers' torch fallbacks. It gives coherent,
  highly repetitive TinyStories text, which is good for ngram/suffix acceptance.
  Output: `experiments/pyt_qwen3next_spd/m70_hf.json`.

### 2026-11-20 — Phase 2, run 1: tiny-random baseline (no SpD, zero code changes): FAILS, host-side (MEASURED)
```
qwen3_next.py:568 linear_attn -> qwen_gdn_linear_attn.py:963 forward_cuda
  -> torch.ops.vllm.qwen_gdn_attention_core -> _forward_core:1290 (attn_metadata None => profile run)
  -> _warmup_prefill_kernels:1202 torch.accelerator.empty_cache()
RuntimeError: device_allocator INTERNAL ASSERT FAILED at c10/core/CachingDeviceAllocator.h:261 ...
              Allocator for qaic is not a DeviceAllocator.
```
- **Type:** Python/PyTorch runtime error raised **before any GDN Triton kernel launches**. It is neither a
  Triton compile error nor a device error, and it has nothing to do with SpD.
- **Cause:** `profile_run` calls GDN's `_warmup_prefill_kernels`, which ends with the generic
  `torch.accelerator.empty_cache()`. torch_qaic's allocator is not registered as a c10 `DeviceAllocator`, so
  the generic accelerator API asserts.
- vllm_qaic already routes its own calls through `current_platform.empty_cache` / `torch.qaic.empty_cache()`
  (`patch/patch_mem_utils.py`, `patch/patch_parallel_state.py`, `worker/worker.py:333`), but this upstream call
  site is not covered. `mamba_mixer2.py:704` has the same call, so Mamba2 would hit it too once past its kernel.
- **Q1 strict answer: NO**, not with zero code changes. The minimum fix is a shim
  `torch.accelerator.empty_cache -> torch.qaic.empty_cache`, either in vllm_qaic's patch set or in torch_qaic
  (registering a DeviceAllocator).
- Next: rerun with that shim applied in the experiment process only (`_shim_accelerator_empty_cache`, opt out
  with `QWEN3NEXT_NO_EMPTY_CACHE_SHIM=1`), to reach the kernels.

### 2026-11-20 — Phase 2, run 2: tiny-random baseline + empty_cache shim: FAILS, vllm_qaic bug (MEASURED)
```
qwen_gdn_linear_attn.py:856 _output_projection -> self.norm(core_attn_out, z)
  -> vllm_qaic/ops/layernorm.py:108 QAicRMSNormGated.forward_oot
     F.rms_norm(x, [self.hidden_size], self.weight, self.eps)
AttributeError: 'QAicRMSNormGated' object has no attribute 'hidden_size'
```
- **Progress before the failure:** the GDN prefill warmup kernels (`fused_post_conv_prep`,
  `chunk_gated_delta_rule`) compiled and ran on QAIC through `qcom_hexagon_backend` with no error. Their numerics
  are not checked here; that is the kernel probe's job.
- **Cause:** a vllm_qaic bug, not an upstream one. Upstream 0.30.0 `RMSNormGated.__init__` sets only `eps`,
  `activation`, `weight`, `group_size` and `norm_before_gate`; it never sets `hidden_size`. vllm_qaic's OOT
  override (registered in `vllm_qaic/ops/__init__.py:54`) reads it anyway. `main` has the same code. Any PYT
  model that uses `RMSNormGated` is affected (Qwen3-Next GDN, Mamba2).
- **Second required code change for Q1:** use `self.weight.shape[-1]` instead of `self.hidden_size`.
- Next: run 3 adds an experiment-only shim, `_shim_rmsnorm_gated_hidden_size` (a `hidden_size` property returning
  `weight.shape[-1]`; opt out with `QWEN3NEXT_NO_NORM_SHIM=1`).

### 2026-11-20 — Phase 2, run 3: tiny-random baseline + both shims: FAILS, in the MoE kernel, not GDN (MEASURED)
```
qwen3_next.py:257 (sparse MoE block) -> moe_runner -> vllm_qaic/ops/unquantized_fused_moe_method.py:127
  -> vllm_qaic/_custom_ops.py:280 _unquantized_fused_moe_hmx_op
RuntimeError: Device id: 0 QID: 0 Msg: Failed to synchronize stream. Error code: 500
```
- **Type:** a QAIC device runtime error (stream sync code 500), raised during `profile_run`.
- **Progress:** the GDN layers' forward passes (conv, `chunk_gated_delta_rule` prefill, gated norm) all finished
  before the failure.
- **Likely cause (HYPOTHESIS):** tiny-random has `hidden_size=8` and 32 experts with top-10 routing, which is
  probably below the HMX kernel's tiling minimum. The HMX routed-MoE op is model-agnostic vllm_qaic code, so this
  is not a GDN finding. Caveat: device errors can surface late, so an earlier asynchronous kernel fault cannot be
  fully excluded.
- **Decision:** move to the trained 70M model (`hidden_size=256`, 16 experts, top-2), which has realistic shapes.
  Retrying tiny-random with `QAIC_UNQUANTIZED_FUSED_MOE_KERNEL=hvx` is optional follow-up.

### 2026-11-20 — Phase 2, run 4: 70M baseline + empty_cache and norm shims: passes profile_run, FAILS at KV-cache setup (MEASURED)
```
core.py:319 _initialize_kv_caches -> kv_cache_utils.get_kv_cache_groups
  -> unify_hybrid_kv_cache_specs -> _promote_local_kv_cache_specs
ValueError: Failed to promote local KV cache specs to one unified type.
```
- **Progress:** `profile_run` finished, so the full 70M forward pass ran on QAIC, including the GDN layers, gated
  RMSNorm and the HMX MoE kernel at hidden size 256. This supports reading run 3 as a shape problem with
  tiny-random.
- **Cause:** `vllm/config/vllm.py:1988` sets `disable_hybrid_kv_cache_manager=True` whenever
  `current_platform.support_hybrid_kv_cache()` is False. `QaicPlatform` inherits the `Platform` default, False;
  CUDA, ROCm, XPU and CPU all return True. With the hybrid KV-cache manager disabled, vLLM tries to fold every
  layer into one spec type. That works for sliding-window plus full attention, but it cannot fold `MambaSpec`
  into `FullAttentionSpec`, so it raises.
- **Third required code change for Q1:** `QaicPlatform.support_hybrid_kv_cache()` must return True, at least in
  PYT mode. Experiment shim: `_shim_platform_hybrid_kv_cache` (opt out with `QWEN3NEXT_NO_HMA_SHIM=1`). The AOT
  implications are not assessed.

### 2026-11-20 — Phase 2, run 5: 70M + hybrid-KV shim: KV cache allocated, FAILS in vLLM's Qwen GDN JIT warmup (MEASURED)
- KV-cache setup now works: 27.92 GiB available, and the KV-cache groups are `[512, 512, 512, 80]`, i.e. GDN
  state groups plus an attention group.
- `compile_or_warm_up_model` -> `kernel_warmup` -> `qwen_triton_warmup` ->
  `_warm_fused_post_conv_kernel` -> Hexagon launcher:
  `RuntimeError: ... Failed to throttle stream submission. Error code: 500`.
- **Localized (QID 10, mode `QWEN3NEXT_WARMUP=localize`):** syncing after each warmup sub-step shows
  - `gated_rms_norm` OK;
  - **`causal_conv1d_fwd` faults the device** (`Failed to synchronize stream. Error code: 500`).
  The warmup calls `causal_conv1d_fn` on a 1-token dummy input with `cache_indices=[NULL_BLOCK_ID]`, an empty
  `has_initial_state` and `metadata=None`. The `fused_post_conv` error in the unlocalized run was the *next*
  submission after the conv fault, not the faulting kernel.

### 2026-11-20 — Phase 2, run 6: 70M with the warmup skipped: FAILS on the first real prefill (MEASURED)
- `init engine` finished (27.8 s). On the first `generate()` prefill, the GDN `_forward_core` path compiles
  `_causal_conv1d_fwd_kernel`. The next submission, `fused_post_conv_prep` (`qwen_gdn_linear_attn.py:1441`), then
  fails with `Failed to throttle stream submission. Error code: 500`.
- Reading (HYPOTHESIS, matches run 5's localization): `causal_conv1d_fn` faults the device on the real prefill
  inputs too. Both runs show the same 2D-pointer warning, `Cannot transform tensor of pointers into a single base
  pointer with tensor of offsets`, and so does the kernel probe, which runs this kernel *without* a fault. The
  warning alone is therefore not the cause; the arguments differ between the two cases. Run 7 traces the
  arguments and syncs around each call.
- Run 6 also covers baseline2 (same failure), so the failure is deterministic.

### 2026-11-20 — Phase 1 kernel probe (QID 9), INTERIM results from the subagent logs
| Stage | Kernel(s) | QAIC fp16 vs CPU fp32 |
|---|---|---|
| conv_fn | `causal_conv1d_fn` prefill, 2 reqs, probe-built metadata | OK (rel ~4e-4) |
| decode | `causal_conv1d_update` + recurrent decode | OK |
| spec | `causal_conv1d_update(IS_SPEC_DECODING)`, `fused_sigmoid_gating` with 2D indices | OK: slot t holds the state after tokens 0..t; unused slots untouched |
| chain | two SpD steps, step 2 reads slot a-1 | OK |
| **chunk** | `chunk_gated_delta_rule` (FLA prefill) | **NaN** output and final state; `post_conv` and the `u` sub-step are OK, `h[...,0]`=initial state is OK |
| **hazard** | zero-draft (non-spec) decode after a step with 3 accepted tokens | **STALE**: reads `block_table[0,0]`, the post-token-0 state; output matches the stale ref to 3e-6 and is 7e-3 away from the correct ref |
(Final subagent report pending; see the subagent section when it lands.)

### 2026-11-20 — Phase 2, run 7: conv1d argument trace: device fault CONFIRMED in `causal_conv1d_fn` (MEASURED)
- **Experiment:** wrap the GDN module's `causal_conv1d_fn`, read every input back to the host, call
  `torch.qaic.synchronize()`, run the kernel, then sync again.
- **Result:** pre-sync is OK and **post-sync fails** (`Failed to synchronize stream. Error code: 500`), so the
  fault is inside this kernel.
- The inputs look valid:
  - x is `(512, 142)` (a transposed view);
  - `cache_indices` is `[1, 5, ..., 29]`;
  - `query_start_loc` is `[0, 27, ..., 142]`;
  - `has_initial_state` is all False.
- The difference from the kernel probe, where the same kernel runs cleanly, is **`conv_states`**: a view of
  shape `(182991, 512, 3)` with strides `(40960, 1, 512)` over the shared hybrid KV pool, roughly 15 GB of fp16.
  The probe used a small pool.

### 2026-11-20 — Phase 2, runs 8 and 9: KV pool capped at 1 GiB: the device fault is GONE, but the output is garbage (MEASURED)
- `kv_cache_memory_bytes=1 GiB` gives `conv_states` shape `(6553, 512, 3)`, same strides. All three traced conv
  calls pass post-sync, and generation finishes: 189 steps, 8 prompts × 64 tokens.
- **Root cause of the device faults (CONFIRMED by this A/B test; the mechanism is HYPOTHESIS):**
  - the GDN Triton kernels fault when the state-pool tensor they get is very large;
  - Hexagon is a 32-bit DSP, so a tensor of about 15 GB probably can't be mapped or addressed by a kernel;
  - the vllm_qaic attention and MoE ops don't hit this because they are not handed the whole pool as one Triton
    pointer.
  - The warmup fault in run 5 is very likely the same thing: the warmup's `causal_conv1d_fn` received the real
    `conv_state` pool.
- **Fourth required change for Q1:** keep each GDN state pool kernel-addressable, e.g. cap or partition the
  hybrid KV pool on QAIC.
- **Output:** both runs produce **one constant token for every prompt and every step**, with 0/64 positional
  agreement with HF fp32 on all 8 prompts.
  - Run 8 (no pad patch) emits `持平` repeatedly.
  - Run 9 (with the zero-pad patch for `recompute_w_u_fwd_kernel`) emits `-col` repeatedly.
  - The pad patch changes the symptom, so the chunk-prefill NaN contributes, but something else is also wrong.
  - Next: a per-sublayer activation diff against HF fp32 on one prompt (`layer_diff.py`).

### 2026-11-20 — Phase 1 kernel probe: FINAL (subagent; QID 9; 70M shapes, fp16, vs own CPU fp32 refs)
| Stage | Result |
|---|---|
| 1 `causal_conv1d_fn` | OK (rel 3.9e-4; conv state exact; other slots untouched) |
| 2 chunk prefill (unmodified) | **NaN** whenever a sequence is not a multiple of 64 tokens (lens 40 and 24). 64/64 is OK (≤5.5e-4) |
| 2' chunk + zero-pad patch | OK (≤8.9e-4) |
| 3 plain decode (packed, and `fused_sigmoid` 1D) | OK (≤9.9e-4) |
| 4a `causal_conv1d_update(num_accepted)`, a=1..4 | OK; the buffer shifts by a-1 exactly |
| 4b `fused_sigmoid` with 2D indices, a=1..4 | OK; reads slot [n,a-1], slot [n,t] = state after tokens 0..t. A control shows that reading [n,0] would be off by 6e-3 to 2.5e-2, so the test can tell slots apart |
| 4c chain of two spec steps | OK (≤1.1e-3) |
| 4d **zero-draft hazard** | **STALE on all 3 paths** (packed decode, 1D `fused_sigmoid`, mixed-batch prefill path) |

- **Chunk NaN root cause (CONFIRMED by single-load ablation):**
  - in `third_party/flash_linear_attention/ops/wy_fast.py`, `recompute_w_u_fwd_kernel` loads `b_k` with
    `tl.load(p_k, boundary_check=(0, 1))` and no `padding_option`;
  - on QAIC the out-of-bounds lanes hold garbage or NaN bits, and `0 × NaN` in `tl.dot(b_A, b_kb)` poisons
    `w`;
  - the NaN count varies from run to run (2064 vs 16384), which fits uninitialized memory;
  - adding `padding_option="zero"` to the k load alone fixes it;
  - other unpadded boundary-checked FLA loads (in chunk_delta_h, chunk_o, kkt, and `v`) are at risk but passed
    the probe.
- **The 4d hazard in code (subagent citations, 0.30.0):**
  - in `mamba_cache_mode="none"` (forced by QAIC at `vllm_qaic/platform.py:495`, and by `config.py:651-652`
    whenever prefix caching is off), **nothing ever copies or rotates slot a-1 into slot 0**:
    `gpu_model_runner.py:1605` only computes `num_accepted_tokens`, `attention/backends/utils.py:1162-1163`
    leaves the block table unchanged, and the MambaManager never rotates blocks;
  - a zero-draft request becomes `-1` (`ngram_proposer.py:126-131`, `suffix_decoding.py:54,61`,
    `scheduler.py:812-823`, `gpu_model_runner.py:2259-2279`) and reads `block_table[:,0]` with
    `num_accepted=None` (`gdn_attn.py:230-261`);
  - in a mixed batch it is reclassified as a prefill and still reads `[mask,0]` (`gdn_attn.py:286-290, 330-332`);
  - the stale state then persists, because the next spec step uses num_accepted=1 relative to the corrupted
    slot 0.
  - Only "align" mode copies, and only at block boundaries (`v1/worker/mamba_utils.py`).
- **Probe caveats:** the references are the subagent's own fp32 implementations; the shapes are 70M-only; the
  methodology was self-reviewed; bf16 takes an untested fused custom-op decode path
  (`qwen_gdn_linear_attn.py:916-930`); and whether upstream CUDA hits the same hazard was not verified E2E.

### 2026-11-20 — Phase 2, run 10: per-sublayer activation diff vs HF fp32 (prompt 0, prefill): root cause of the garbage output is in the GemmaRMSNorm residual path (MEASURED)
`layer_diff.py`: forward hooks on embed and each sublayer of both models; vLLM fp16 on QAIC (with zero-pad and
1 GiB KV) is compared against HF CPU fp32. Excerpt (`rel = max|Δ| / max|ref|`):

| module | which | rel |
|---|---|---|
| embed_tokens | out | 0 |
| layers.0.input_layernorm (no residual) | out | 5.4e-4 |
| layers.0.linear_attn (**GDN: conv + chunk prefill + gated norm**) | out | **1.2e-3 OK** |
| layers.0.post_attention_layernorm (**with residual**) | out | **5.0e-1** |
| every later module | out | 0.7 to 1.0 |

(The `in` rows of the residual norms aren't comparable: vLLM passes `(x, residual)` separately, while HF passes
the sum.)

- **Root cause (CONFIRMED by reading the code; the post-fix diff is pending):** `vllm_qaic/ops/layernorm.py`
  `QAicGemmaRMSNorm.forward_oot` passes the raw `self.weight` to the Hexagon fused add+RMSNorm kernel when a
  residual is given.
  - GemmaRMSNorm stores a zero-centred weight, so the effective scale is `1 + weight`.
  - The no-residual branch in the same method correctly uses `1.0 + self.weight`.
  - Qwen3-Next aliases `GemmaRMSNorm as Qwen3NextRMSNorm` for every input, post-attention and final norm.
  - **Same code on `main`.** This is a vllm_qaic bug that affects **every** PYT model using GemmaRMSNorm with a
    residual (Gemma 1/2/3, Qwen3-Next); it is not specific to GDN.
- **Fifth required change for Q1:** pass `1.0 + self.weight` in the residual branch. Experiment shim:
  `_shim_gemma_rmsnorm_residual` (opt out with `QWEN3NEXT_NO_GEMMA_SHIM=1`).
- **The GDN layer itself matches HF at fp16 tolerance on QAIC** once the zero-pad patch and KV cap are in place.

### 2026-11-20 — Phase 2, run 11: layer diff with the Gemma residual-norm shim: layers 0-2 fixed, next divergence is in full attention (MEASURED)
| module | out rel vs HF fp32 |
|---|---|
| layers.0-2 (GDN `linear_attn`, residual norms, MoE) | 5e-4 to 3.4e-3 (**OK**, fp16 tolerance) |
| layers.3.input_layernorm | 2.2e-3 |
| **layers.3.self_attn** (full attention, head_dim 256, 2 q / 1 kv heads, partial NeoX RoPE with rotary_dim 64, sigmoid output gate) | **3.7e-1** |
| everything after it | 0.26 to 0.45 |

- The 70M baseline E2E run with all shims is still 0/64 positional match on every prompt, consistent with the
  remaining attention error.
- Upstream's fused `fused_qk_rmsnorm_rope_gate` is CUDA-only, so QAIC takes the eager path: split, then
  q_norm/k_norm (GemmaRMSNorm without residual, OK), then `QAicRotaryEmbedding` (rotary_dim=64 of 256, so
  `query[..., :64]` is a **non-contiguous** slice passed to `torch.ops.qaic.rotary_embedding`), then the QAIC
  attention backend, then `* sigmoid(gate)`.
- Next: `attn_probe.py` captures layer-3 rotary and attention inputs/outputs on QAIC and recomputes each on CPU.

### 2026-11-20 — Phase 2, run 12: layer-3 attention probe and RoPE microtest: QAIC rotary op is wrong on strided multi-head input (MEASURED, CONFIRMED)
`attn_probe.py` captured layer 3's `rotary_emb` and `attn` inputs/outputs on QAIC and recomputed each on CPU:

| piece | rel vs CPU ref |
|---|---|
| RoPE q (2 heads), rotated part | **1.07** |
| RoPE q, pass-through part | 0 |
| RoPE k (1 head) | 6.3e-4 |
| `QAicAttentionBackendImpl`, causal GQA prefill, head_dim 256 (on the captured q, k, v) | 6.0e-4 (OK) |

Note: this checkpoint's attention is so sink-dominated that the non-causal reference is equally close, so
the probe can't separately confirm the causal mask.

`rope_micro.py` called `torch.ops.qaic.rotary_embedding` directly on a `[T, H, 256][..., :64]` slice:

| H | strided slice | `.contiguous()` |
|---|---|---|
| 1 (stride (256, 256, 1)) | 3.8e-4 | 3.8e-4 |
| 2 (stride (512, 256, 1)) | **1.06** | 3.8e-4 |
| 4 (stride (1024, 256, 1)) | **1.80** | 6.0e-4 |

- **Root cause (CONFIRMED):** the QAIC rotary op ignores the head stride of non-contiguous input, despite the
  docstring promising a CPU fallback for non-contiguous inputs. `QAicRotaryEmbedding.forward_static` passes
  `query[..., :rotary_dim]`, which is strided whenever `rotary_dim < head_size`.
- **Same code on `main`.** This affects every PYT model with partial rotary and more than one head (Qwen3-Next,
  Phi, GLM, StableLM, GPT-NeoX, ...). Full-rotary models are unaffected, because the slice is then contiguous.
- **Sixth required change for Q1:** call `.contiguous()` on the rot slice, or fix the op's stride handling.
  Experiment shim: `_shim_rope_contiguous` (opt out with `QWEN3NEXT_NO_ROPE_SHIM=1`).

### 2026-11-20 — Phase 2, run 13: layer diff with the RoPE shim: all layers match HF; remaining divergence is the untied lm_head (MEASURED)
- Every module from embed to the final `norm` is within **rel ≤ 4.9e-3** of HF fp32. That covers all 6 GDN
  layers, both full-attention layers and all MoE blocks.
- vLLM's first token is still 117763, while HF's argmax is 1486. The 70M E2E baseline is still 0/64, and
  baseline vs baseline2 is **nondeterministic** (first divergence 0 to 4).
- **Cause (CONFIRMED by inspection):** `config.tie_word_embeddings=true`, and the safetensors file has
  `model.embed_tokens.weight` but **no** `lm_head.weight`. Upstream `Qwen3NextForCausalLM` (0.30.0) always
  builds a separate `ParallelLMHead` and never ties it, so lm_head stays uninitialized memory. That explains both
  the garbage and the run-to-run nondeterminism.
  - This is a mismatch between the upstream model and this community checkpoint, **not a QAIC bug**: official
    Qwen3-Next checkpoints ship an untied lm_head.
  - Experiment shim: after `LLM()`, `apply_model` copies `embed_tokens.weight` into `lm_head.weight` (opt out
    with `QWEN3NEXT_NO_TIE_SHIM=1`).

### 2026-11-20 — Phase 2, runs 14-16: all shims incl. lm_head tie: partially correct and NONDETERMINISTIC (MEASURED)
Config: 70M, fp16, 1 GiB KV, the wy_fast zero-pad patch, and the empty_cache, gated-norm, Gemma-residual,
hybrid-KV, RoPE-contiguous and lm_head-tie shims. Positional match vs HF fp32 (64 tokens per prompt):

| run | max_num_seqs | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---|---|---|---|---|---|---|---|---|
| m70t baseline | 8 | **64** | 3 | 0 | 0 | 1 | 0 | 1 | 0 |
| m70spd baseline (identical config) | 8 | 1 | **64** | 1 | 0 | 1 | 11 | 1 | 1 |
| m70spd1 baseline | 1 | **64** | 1 | **64** | 1 | **64** | 11 | **64** | **64** |

(Prompt lengths 27/22/14/15/18/16/14/16.)

- At **mns=8, identical configs disagree with each other**: m70t baseline vs baseline2 match 0/64 on 7 of 8
  prompts. Even the prefill token is nondeterministic, so some kernel reads uninitialized memory in batched
  (varlen) execution. That makes the mns=8 SpD comparisons (m70spd ngram/suffix) **uninterpretable**; they are
  discarded.
- At **mns=1**, 5/8 prompts match HF exactly for 64 tokens. p1 and p3 collapse to a repeated token (`jump`)
  from the **first decode step**. p5 diverges at token 11, which may be a genuine fp16-vs-fp32 near-tie
  (unverified).
- **Hypothesis:** the same class of bug as the wy_fast k-load. Masked or boundary-checked `tl.load`s without
  `other`/`padding_option` read garbage on QAIC, in other FLA/GDN kernels: varlen chunk prefill with more than
  one sequence, and the decode kernels.
- Test: `_shim_tl_load_zero` (`QWEN3NEXT_TL_LOAD_ZERO=1`) wraps the `tl.load` builtin globally to force
  `padding_option="zero"` and `other=0`, run with a fresh `TRITON_CACHE_DIR`.

### 2026-11-20 — Phase 2, runs 17-18: global tl.load zero-fill: deterministic, and matches HF up to fp16 near-ties (MEASURED) — **Q1 RESOLVED**
Same config as runs 14-16, plus `QWEN3NEXT_TL_LOAD_ZERO=1` and a fresh `TRITON_CACHE_DIR`; mns=8.

| check | result |
|---|---|
| baseline vs baseline2 | **64/64 on all 8 prompts** (deterministic; before: 0/64 on 7/8) |
| baseline vs HF fp32 | 64/64 on p0-p4 and p7; p5 diverges at 11, p6 at 27 |
| HF top-2 margin at p5@11 / p6@27 | 0.0172 / 0.0039 logits; in both cases vLLM chose HF's **rank-2** token (`margin_check.py`) |

- **Root cause of the remaining nondeterminism (CONFIRMED by this A/B; the exact kernel(s) are not
  isolated):** Triton loads in the GDN path that don't specify `other` (masked) or `padding_option`
  (boundary-checked) get **uninitialized values in out-of-range lanes on the QAIC Hexagon backend**.
  - Upstream relies on those lanes being harmless or zero on GPU.
  - The wy_fast k-load is one confirmed instance; at least one more is in the varlen-batched and decode paths.
  - **Seventh required change for Q1:** either the Hexagon Triton backend zero-fills out-of-range lanes (best:
    a single fix in the backend), or the FLA/GDN kernels get explicit `other=0`/`padding_option="zero"`.
- The two remaining divergences are fp16-vs-fp32 near-ties, i.e. expected precision drift, not a bug.

### 2026-11-20 — Runs 19-21: Q2 E2E with the Q1-deterministic config (all Q1 shims + zero-fill)
Config: `--bypass-hybrid`, `QWEN3NEXT_TL_LOAD_ZERO=1`, `QWEN3NEXT_ZERO_PAD=1`, KV 1 GiB, k=4, 8 prompts × 64
greedy tokens, `ignore_eos`. Reference = the vLLM non-SpD baseline from the same run (not HF).

**Baseline determinism control (run 19, `m70z1`, mns=1, QID 11):** baseline vs baseline2 is 64/64 on all 8
prompts. vs HF fp32, 7/8 are exact over 64 tokens; p5 diverges at 11, the measured fp16 near-tie. The mns=8
baseline was already deterministic (run 18).

**SpD vs baseline (runs 20 `m70zspd1` mns=1 QID 12; run 21 `m70zspd` mns=8 QID 8):**

| run | method | drafts / accepted | prompts identical to baseline | first-divergence index per prompt |
|---|---|---|---|---|
| mns=1 | ngram | 671 / 130 | 1/8 (p7) | 9, 14, 43, 18, 29, 11, 36, — |
| mns=1 | suffix | 499 / 228 | 2/8 (p2, p6) | 9, 8, —, 19, 4, 11, —, 19 |
| mns=8 | ngram | 727 / 165 | 0/8 | 5, 2, 5, 2, 6, 5, 5, 3 |
| mns=8 | suffix | 477 / 196 | 0/8 | 4, 2, 3, 2, 4, 3, 3, 3 |

Acceptance is nonzero, so equality wasn't vacuous. **Q2 answer so far: NO. Native state caching does not give
greedy output identical to non-SpD for ngram/suffix on QAIC.**

**Probe-summarizer bug (fixed offline):** `summarize_probe` reported `hazard_rows=0` at mns=1. When no request in
the batch has drafts, the runner passes `num_decode_draft_tokens_cpu=None`, so the probe logs `draft=None` and the
summarizer skipped exactly the zero-draft steps. New offline analyzer `experiments/pyt_qwen3next_spd/hazard_analyze.py`:
- tracks each request's computed-token count `C = seq_len - qlen` across steps;
- infers accepted `a = C_next - C`;
- flags a hazard when a spec step with `a > 1` is followed by a non-spec qlen-1 decode. That decode reads
  `block_table[:,0]` (slot 0), but the valid state is in slot `a-1`. The first output it may corrupt is
  index `C_next - prompt_len + 1`.
- Cross-check: the probe's `nat_raw` at every spec step equals the trajectory-implied `a` (mns=8: 0/549 and
  0/711 mismatches). So the runner's `num_accepted_tokens` bookkeeping is correct. The 2/505 at mns=1 are
  artifacts of a block-id reuse dropping one probe record.

**H1 (zero-draft stale slot) at mns=1: strongly correlated.**

| method | prompt | first_div | first hazard output idx | hazards |
|---|---|---|---|---|
| ngram | 0 | 9 | 9 | 2 |
| ngram | 1 | 14 | 14 | 4 |
| ngram | 2 | 43 | 36 | 3 |
| ngram | 3 | 18 | 16 | 5 |
| ngram | 4 | 29 | 29 | 1 |
| ngram | 5 | 11 | 9 | 2 |
| ngram | 6 | 36 | 36 | 2 |
| ngram | 7 | — | — | 0 |
| suffix | 0 | 9 | 9 | 2 |
| suffix | 1 | 8 | 7 | 2 |
| suffix | 2 | — | — | 0 |
| suffix | 3 | 19 | 19 | 3 |
| suffix | 4 | 4 | 4 | 1 |
| suffix | 5 | 11 | — | 0 |
| suffix | 6 | — | — | 0 |
| suffix | 7 | 19 | 19 | 1 |

- Every hazard-free request matches baseline exactly, except suffix p5, which diverges at index 11. That is the
  same token where the baseline itself sits on a 0.017-margin fp16 near-tie vs HF. It is plausibly a spec-path
  numeric flip, not a state bug.
- No request diverges *before* its first hazard. When divergence lags the hazard (by 1-7 tokens), the stale
  state perturbed the logits without immediately flipping the argmax.

**mns=8: divergence starts earlier than the first hazard (output idx 2-6, before any hazard in 14/16
requests).** A second mechanism, H2, appears only in batches that mix spec and non-spec rows:
- upstream `gdn_attn.py` (~l.282-290) reclassifies non-spec qlen-1 decodes as **prefills** whenever the batch
  also has spec decodes ("the prefill kernel handles 1-token sequences with initial state correctly");
- so in a mixed batch those rows run `causal_conv1d_fn` + `chunk_gated_delta_rule` with an initial state,
  instead of the decode kernels;
- the probe shows this from the first decode step (e.g. step 3: `q=[1,1,1,1,1,1,5,5]`, `ns=2`, `np=6`);
- the stale-slot read (H1) also applies on this route (`prefill_state_indices = block_table[:,0]`).

H2 was first a hypothesis; the kernel microtest below CONFIRMS it at kernel level.

**Causal control (run 22, `m70pad*`, QIDs 13/14; results below):** `QWEN3NEXT_PAD_DRAFTS=1` pads every ngram/suffix
proposal to exactly k=4 drafts. Then no decoding row ever takes the non-spec path, and both H1 and H2 are removed.
Padded drafts are verified normally, so greedy output must equal baseline if the spec-path kernels are correct.

### 2026-11-20 — Run 22: padded-draft causal control (MEASURED)
`QWEN3NEXT_PAD_DRAFTS=1` wraps `NgramProposer.propose` / `SuffixDecodingProposer.propose` and pads each
proposal to k=4 by repeating the last draft (or the last sampled token). The probe confirms 0 non-spec decode
rows and 0 mixed steps, so neither H1 nor H2 can occur. Same config as runs 19-21 otherwise.

| run | method | drafts / accepted | identical to baseline | divergences (prompt@idx) |
|---|---|---|---|---|
| mns=1 | ngram | 1420 / 153 | 5/8 | p1@17, p5@11, p6@27 |
| mns=1 | suffix | 1160 / 217 | 6/8 | p5@11, p6@27 |
| mns=8 | ngram | 1420 / 153 | 6/8 | p1@17, p5@11 |
| mns=8 | suffix | 1128 / 228 | 7/8 | p6@27 |

Unpadded, for comparison (runs 20/21): 1/8, 2/8 at mns=1; 0/8, 0/8 at mns=8.

HF fp32 top-2 margin at each remaining divergence (`margin_check.py`): p1@17 = 0.0092, p5@11 = 0.0172,
p6@27 = 0.0039. In every case one run picks HF's rank-1 token and the other the rank-2 token. These are
fp16 near-ties. The same three positions recur across methods and batch sizes.
Reading: removing the non-spec path removes almost all divergence. The spec-path kernels (2D state indices,
`num_accepted_tokens` slot selection, IS_SPEC_DECODING conv update) are correct up to fp16 near-ties.

### 2026-11-20 — H2 kernel microtest: `causal_conv1d_fn` short-sequence conv-state write is wrong on QAIC (MEASURED, CONFIRMED)
Scripts: `experiments/pyt_qwen3next_spd/h2_prefill_as_decode_probe.py`, `h2_conv_state_diag.py` (+ `.log`,
`h2_conv_state_diag_noshim.log`). These run the mixed-batch "1-token decode as prefill with initial state"
route on QAIC, fp16, vs a CPU fp32 reference.
- 117 checks, 13 mismatches. **All the mismatches are in the conv-state write-back.** The conv output and the
  `chunk_gated_delta_rule` output/SSM state are correct.
- Pattern (width 4, state window 3): for a 1-token sequence, expected `[old1, old2, x0]`, device writes
  `[old1, old2, 0]`. For a 2-token sequence the device writes `[old2, 0, 0]`. Sequences with ≥3 tokens are
  correct. The faulty code is the `seqlen < state_len` with-initial-state branch of `_causal_conv1d_fwd_kernel`
  (~l.244-283 in `layers/mamba/ops/causal_conv1d.py`).
- The same result occurs with and without the `tl.load` zero-fill shim. The compiler warns "Cannot transform
  tensor of pointers into a single base pointer" on this branch. This looks like a Hexagon Triton miscompile
  of the gathered-pointer store rather than an upstream logic bug (hypothesis: not run on GPU; the CPU reference
  implementing the documented semantics gives the expected window).
- Effect in E2E: the next decode reads a conv window with zeros in it. This explains the mns=8 divergences
  that begin before any H1 hazard.

### 2026-11-20 — Code check (read-only agent, vLLM 0.30.0): H1 is an unacknowledged upstream gap
- `mamba_cache_mode="none"` (QAIC PYT, prefix caching off): nothing copies slot a-1 into slot 0 for a request
  that has no drafts this step. The non-spec path reads `block_table[:,0]` with `num_accepted_tokens=None`.
- `"align"` (GPU default with prefix caching): same hazard, unless the step crosses a mamba block boundary,
  which triggers the slot copy. `"all"` is rejected for Qwen3-Next.
- By contrast, the Mamba/Mamba2 metadata builder keeps every decode row on the multi-slot path, which makes it
  immune to this.
- Async scheduling pads drafts with -1 placeholders, which would avoid it, but async is disabled for CPU
  ngram/suffix proposers.
- No code comment, doc or test covers zero-draft requests for GDN. MTP/EAGLE always propose k drafts, so they
  never hit it.

## Methodology review (independent subagent, 2026-11-20): **methodology accepted with limits**
Checked facts:
- mns=1 probes are all single-row, so there is no H2 at mns=1.
- mns=8 unpadded runs are mixed (135/156 and 102/141 steps); padded runs have 0 mixed steps.
- `kernel_probe_hazard.log` shows the non-spec output matches the stale-state reference (2.9e-6), not the
  correct one (6.9e-3).

Supported:
- **(A)** Native caching as shipped is not greedy-identical for ngram/suffix, for this model and config.
- **(B, H1)** The zero-draft stale-slot gap exists in source and at kernel level, and is the probable cause at
  mns=1.
- **(C), weakly:** under padded, all-spec q=5 batches, there is no divergence a margin check can't explain.

Not supported:
- That H2 is QAIC-specific.
- That H1 and H2 are independent causes, separated end to end.
- That the gap manifests on GPU at runtime.
- That the spec path is exact up to fp16 noise. The near-tie explanation was applied after the fact, using HF
  margins rather than vLLM's.
- Generalization beyond one 70M model, 8 prompts, k=4, and an 8-shim engine.

Open controls (not run):
1. A targeted H1 fix at mns=1 without padding (copy slot a-1 → slot 0, or keep zero-draft rows on the spec
   path). Expect 8/8 identity apart from logit-verified ties.
2. An H2-only fix at mns=8. Divergence should then move to the H1 hazards.
3. vLLM logprobs in both runs, comparing per-position logits.
4. Run `causal_conv1d_fn` under triton-cpu or `TRITON_INTERPRET=1`, to settle whether H2 is specific to QAIC.
5. The acceptance-count histogram under padding.
6. More prompts or models, and a GPU run.

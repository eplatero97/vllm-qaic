# Qwen3-Next (Gated DeltaNet) on QAIC PYT eager: experiment archive and reproduction guide

This directory holds every script, log and result from the 2026-11-20 investigation into two questions:

1. **Q1:** does Qwen3-Next run end to end in PyTorch-eager (PYT) mode on one QAIC QID (fp16, greedy)?
2. **Q2:** does upstream vLLM's native recurrent-state caching for speculative decoding (SpD) give greedy
   output identical to non-SpD when the proposer is ngram or suffix?

The conclusions, and how each one was reached, are in
[`docs/qaic/pyt_qwen3_next_gdn_spd_findings.md`](../../docs/qaic/pyt_qwen3_next_gdn_spd_findings.md).
Read its "How each conclusion was reached" section first. This README is the lab notebook: what each script
does, the exact command behind each numbered run, and which output files came from it.

> **No `vllm_qaic/` source was changed.** Every fix and workaround below is a monkeypatch ("shim") applied
> inside the experiment process by `run_qwen3next_spd.py`. This branch is based on PR #106
> (`pyt-ngram-suffix-spd`, `14ee2a9`), the only branch where ngram/suffix SpD is allowed in PYT mode.
> The scripts were committed exactly as they were run (not lint-cleaned), so they match the results.

## Environment
- **Checkout and venv:** the PR #106 checkout with its editable `.venv_eager`. That venv has vLLM 0.30.0,
  transformers 5.23, Qualcomm triton 3.7.0 (`qcom_hexagon_backend`), arctic_inference and a built
  `hexagon_kernels.so`.
- **Models:** downloaded into `HF_HOME=/local/mnt/workspace/eplatero/hf`.
  - `theo77186/Qwen3-Next-70M-TinyStories` (`M70`): trained, used for every result.
  - `tiny-random/qwen3-next-moe` (`TR`): random weights, crash test only.
- **Env file:** every device command starts with `source pyt_env.sh`. This directory holds a copy of the
  `/tmp/pyt_env.sh` that was used:
  ```bash
  export HEXAGON_TOOLS=/prj/crd/austin/validation/scratch/users/eplatero/pytorch/pytorch_build_tools/hexagon_tools-21.0.02/Tools
  export HEXAGON_SDK_ROOT=/prj/crd/austin/validation/scratch/users/eplatero/pytorch/pytorch_build_tools/hexagon_sdk-6.5.0
  export HEXAGON_ARCH_VERSION=68
  export QAIC_NUM_CORES=16 QAIC_NUM_THREADS=4 QAIC_DEBUG=1
  export HF_HOME=/local/mnt/workspace/eplatero/hf
  ```
- **QIDs:** choose a free one with `qaic-util | grep -i "qid\|status\|board serial"`. Every run uses one QID.

## Scripts
| Script | Device? | Purpose |
|---|---|---|
| `run_qwen3next_spd.py` | yes | Main E2E harness. `driver <model> <tag> <methods...>` runs each method (`baseline`, `baseline2`, `ngram`, `suffix`) in its own subprocess. Writes `<tag>_<method>.{json,log,json.probe.jsonl}` and `<tag>_summary.json`. `child hf <out.json> <model>` produces the HF CPU fp32 reference. It also hosts all the shims and the GDN metadata probe. |
| `gdn_kernel_probe.py` | yes | Phase 1: runs each GDN kernel stage (`conv_fn`, `chunk`, `decode`, `spec`, `chunk_parts`, `chain`, `hazard`) at 70M shapes, fp16 on QAIC, against CPU fp32 references. |
| `wy_fast_zero_pad.py` | yes | Replaces FLA `recompute_w_u` with a version using `padding_option="zero"`. Imported when `QWEN3NEXT_ZERO_PAD=1`. |
| `layer_diff.py` | yes / CPU | Diffs per-sublayer activations, vLLM-on-QAIC vs HF fp32 (`hf`, `vllm`, `cmp` subcommands). Its `.pt` dumps are not committed; they are regenerable. |
| `attn_probe.py` | yes | Breaks down the layer-3 full-attention divergence (q/k norm, RoPE, SDPA). |
| `rope_micro.py` | yes | Microtest of `qaic.rotary_embedding` on contiguous vs strided multi-head input. |
| `compare_tokens.py` | no | Positional token match and first-divergence index between two result JSONs. |
| `margin_check.py` | CPU | At each divergence, prints HF fp32's top-2 logit margin and where the vLLM token ranks under HF. Run with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. |
| `hazard_analyze.py` | no | Rebuilds per-request accepted counts from a probe log and flags H1 hazard steps. A hazard is a spec step that accepted more than one token, followed by a non-spec 1-token decode. Usage: `hazard_analyze.py probe.jsonl run.json baseline.json`. |
| `h2_prefill_as_decode_probe.py` | yes | H2 microtest: 1- and 2-token sequences with initial state through `causal_conv1d_fn` + `chunk_gated_delta_rule`. This is the mixed-batch route. |
| `h2_conv_state_diag.py` | yes | H2: dumps the conv-state window written by `causal_conv1d_fn` for short sequences. |

### `run_qwen3next_spd.py` switches
| Switch | Default | Effect |
|---|---|---|
| `QWEN3NEXT_NO_EMPTY_CACHE_SHIM=1` | shim on | Turns off the shim mapping `torch.accelerator.empty_cache` to `torch.qaic` (fix 1). |
| `QWEN3NEXT_NO_NORM_SHIM=1` | shim on | Turns off the `QAicRMSNormGated.hidden_size` → `weight.shape[-1]` shim (fix 2). |
| `QWEN3NEXT_NO_HMA_SHIM=1` | shim on | Turns off the shim making `QaicPlatform.support_hybrid_kv_cache()` return True (fix 3). |
| `QWEN3NEXT_KV_BYTES=N` | unset | Sets `kv_cache_memory_bytes` (fix 4). All good runs use `1073741824`. |
| `QWEN3NEXT_WARMUP` | `skip` | `skip`: no-op GDN JIT warmup. `localize`: sync after each warmup sub-kernel. Any other value: upstream warmup. |
| `QWEN3NEXT_NO_GEMMA_SHIM=1` | shim on | Turns off the GemmaRMSNorm residual-path `1 + weight` shim (fix 5). |
| `QWEN3NEXT_NO_ROPE_SHIM=1` | shim on | Turns off the RoPE `.contiguous()` shim (fix 6). |
| `QWEN3NEXT_ZERO_PAD=1` | off | Imports `wy_fast_zero_pad`. Fixes the Phase 1 chunk NaN; superseded by `QWEN3NEXT_TL_LOAD_ZERO=1`. |
| `QWEN3NEXT_TL_LOAD_ZERO=1` | off | Global `tl.load` zero-fill shim (fix 7). Needs a fresh `TRITON_CACHE_DIR`. |
| `QWEN3NEXT_NO_TIE_SHIM=1` | shim on | Turns off tying `lm_head` to `embed_tokens` (fix 8; this is a checkpoint/upstream issue, not QAIC). |
| `QWEN3NEXT_CONV_TRACE=1` | off | Dumps arguments and syncs around `causal_conv1d_fn` (run 7). |
| `QWEN3NEXT_PAD_DRAFTS=1` | off | Pads every ngram/suffix proposal to k=4 drafts (run 22 causal control). |
| `--bypass-hybrid` | off | Hides `is_hybrid` from `QaicPlatform.check_and_update_config` only, so SpD is allowed. The runner still sees `is_hybrid`. |
| `--max-num-seqs N` | 8 | Engine `max_num_seqs` (written "mns" below). |

Fixed engine config: `dtype=float16`, `max_model_len=512`, `enforce_eager`, `async_scheduling=False`, prefix
caching off. For SpD: k=4, ngram lookup 1..4. Sampling: greedy, 64 tokens, `ignore_eos`, 8 TinyStories
prompts.

> **Reproducing early runs:** shims were added one at a time, each after the run that exposed its bug, and
> they are now on by default. To reproduce run N, turn off (`QWEN3NEXT_NO_*`) every shim added after it, and
> for runs 1-5 set `QWEN3NEXT_WARMUP=run`.

## Run index (commands as executed)
Shorthand used in the tables:
- `PY` = `.venv_eager/bin/python -u`; `R` = `run_qwen3next_spd.py`.
- `Z` = `QWEN3NEXT_ZERO_PAD=1`; `KV` = `QWEN3NEXT_KV_BYTES=1073741824`; `TLZ` = `QWEN3NEXT_TL_LOAD_ZERO=1`.
- `Vn` = `QAIC_VISIBLE_DEVICES=n`.

All commands run from this directory unless noted otherwise.

### Phase 0: HF reference (CPU)
| Run | Command | Outputs | Result |
|---|---|---|---|
| HF | `PY R child hf m70_hf.json M70` | `m70_hf.{json,log}` | fp32 reference text used by every comparison |

### Phase 1: GDN kernel probe (QID 9)
All rows use `V9 PY gdn_kernel_probe.py <stage> [x <lens>]`.

| Step | Args / env | Outputs | Result |
|---|---|---|---|
| conv | `conv_fn` | `kernel_probe_conv_fn.log` | OK (rel 3.9e-4) |
| stages | `chunk`, `decode`, `spec` | `kernel_probe_{chunk,decode,spec}.log` | chunk NaN; decode and spec OK |
| parts | `chunk_parts`, `chain`, `hazard` | `kernel_probe_{chunk_parts,chain,hazard}.log` | chain OK. **hazard: the non-spec read matches the stale state (H1 at kernel level).** |
| lengths | `chunk_parts x 40,24` / `x 64,64` | `kernel_probe_chunk_parts_{40_24,64_64}.log` | NaN unless the length is a multiple of 64 |
| pad ablation | `PROBE_ZERO_PAD=1 PROBE_PAD_ONLY={"",k,g,beta} chunk_parts x 40,24` | `kernel_probe_chunk_parts_zeropad_{all,k,g,beta}.log` | Zero-padding the k-load alone fixes the NaN |
| confirm | `PROBE_ZERO_PAD=1 PROBE_PAD_ONLY=k chunk` and `... hazard x prefill1` | `kernel_probe_chunk_zeropad_k.log`, `kernel_probe_hazard_prefill1_zeropad_k.log` | chunk OK; hazard still stale |

### Phase 2: Q1 (runs 1-18)
| Run | Command | QID | Outputs | Result |
|---|---|---|---|---|
| 1 | `V8 PY R driver TR tinyrand baseline` (no shims) | 8 | `tinyrand_baseline.log`, `tinyrand_summary.json` | `torch.accelerator.empty_cache` assert, so fix 1 added |
| 2 | same, tag `tinyrand_shim` | 8 | `tinyrand_shim_*` | `QAicRMSNormGated.hidden_size` AttributeError, so fix 2 added |
| 3 | same, tag `tinyrand_shim2` | 8 | `tinyrand_shim2_*` | MoE HMX stream-sync error 500 (not GDN); switched to M70 |
| 4 | `V8 PY R driver M70 m70 baseline baseline2` | 8 | overwritten by run 6 | Failed to promote KV specs, so fix 3 added |
| 5 | same | 8 | overwritten by run 6 | GDN JIT warmup error 500 |
| 5-loc | `V10 QWEN3NEXT_WARMUP=localize PY R driver M70 m70loc baseline` | 10 | `m70loc_*` | Fault localized to `causal_conv1d_fwd` |
| 6 | same as 4 (warmup now skipped) | 8 | `m70_*` | Fault on the first real prefill |
| 7 | `V8 QWEN3NEXT_CONV_TRACE=1 PY R driver M70 m70trace baseline` | 8 | `m70trace_*` | Fault confirmed in conv1d; `conv_states` view is ~15 GB |
| 8 | `V8 KV QWEN3NEXT_CONV_TRACE=1 PY R driver M70 m70kv1g baseline` | 8 | `m70kv1g_*` | **No fault** (fix 4), but output is one token repeated (0/64) |
| 9 | `V10 KV Z PY R driver M70 m70kv1gzp baseline` | 10 | `m70kv1gzp_*` | Still garbage (0/64) |
| 10 | `PY layer_diff.py hf ld_hf.pt 0`; `V8 Z PY layer_diff.py vllm ld_vllm_zp.pt 0`; `PY layer_diff.py cmp ld_hf.pt ld_vllm_zp.pt` | 8 | `ld_vllm_zp.log` | GemmaRMSNorm residual path rel err 0.5, so fix 5 added |
| 11 | layer_diff as in 10 (`ld_vllm_g`); `V9 Z KV PY R driver M70 m70g baseline` | 8, 9 | `ld_vllm_g.log`, `m70g_*` | Layers 0-2 fixed; layer 3 attention rel 0.37 |
| 12 | `V8 PY attn_probe.py`; `V9 PY rope_micro.py` (stdout only) | 8, 9 | `attn_probe.log` | RoPE wrong on strided input (rel 1.06 at H=2, 1.80 at H=4), so fix 6 added |
| 13 | layer_diff (`ld_vllm_gr`); `V10 Z KV PY R driver M70 m70gr baseline baseline2` | 8, 10 | `ld_vllm_gr.log`, `m70gr_*` | All layers ≤4.9e-3 rel; `lm_head` untied, so fix 8 added |
| 14 | `V8 Z KV PY R driver M70 m70t baseline baseline2` | 8 | `m70t_*` | Partly correct, **nondeterministic** at mns=8 |
| 15 | `V9 Z KV PY R driver M70 m70spd baseline ngram suffix --bypass-hybrid` | 9 | `m70spd_*` | Discarded: the baseline was nondeterministic |
| 16 | as 15, tag `m70spd1`, `--max-num-seqs 1`, QID 10 | 10 | `m70spd1_*` | 5/8 match HF; still not trustworthy |
| 17-18 | `V8 Z KV TLZ TRITON_CACHE_DIR=/local/mnt/workspace/eplatero/triton_cache_tlzero8 PY R driver M70 m70z baseline baseline2` | 8 | `m70z_*` | **Deterministic** (fix 7); 6/8 exact vs HF; the other 2 at fp16 near-ties. **Q1 resolved.** |

### Q2 (runs 19-22)
Every row uses `Z KV TLZ`, plus a `TRITON_CACHE_DIR` under `/local/mnt/workspace/eplatero/`.

| Run | Command | QID | Outputs | Result |
|---|---|---|---|---|
| 19 | `V11 ... PY R driver M70 m70z1 baseline baseline2 --max-num-seqs 1` | 11 | `m70z1_*` | Baseline deterministic 64/64; 7/8 exact vs HF |
| 20 | `V12 ... PY R driver M70 m70zspd1 baseline ngram suffix --bypass-hybrid --max-num-seqs 1` | 12 | `m70zspd1_*` | ngram 1/8, suffix 2/8 identical to baseline |
| 21 | `V8 ... PY R driver M70 m70zspd baseline ngram suffix --bypass-hybrid` | 8 | `m70zspd_*` | ngram 0/8, suffix 0/8 |
| 22 | `QWEN3NEXT_PAD_DRAFTS=1`; `V13 ... driver M70 m70pad1 ngram suffix --bypass-hybrid --max-num-seqs 1` and `V14 ... driver M70 m70pad ngram suffix --bypass-hybrid` | 13, 14 | `m70pad1_*`, `m70pad_*` | 5-7/8 identical; every remaining divergence is an fp16 near-tie |

Notes on runs 19-22:
- Run 22 has no baseline of its own. It is compared against the baselines of runs 20 and 21 (same config,
  deterministic).
- Its background wrapper hit the 2-hour limit after all output files were written.

### H2 microtests (QID 15)
| Step | Command | Outputs | Result |
|---|---|---|---|
| probe | `V15 TRITON_CACHE_DIR=.../triton_cache_h2probe PY h2_prefill_as_decode_probe.py` | `h2_prefill_as_decode_probe.log` | 117 checks, 13 mismatches, all in the conv-state write-back |
| diag | `V15 ... PY h2_conv_state_diag.py` | `h2_conv_state_diag.log` | A 1-token sequence writes `[old1, old2, 0]` instead of `[old1, old2, x0]` |
| diag, no shim | `V15 TRITON_CACHE_DIR=.../triton_cache_h2probe_noshim H2_NO_TL_ZERO=1 PY h2_conv_state_diag.py` | `h2_conv_state_diag_noshim.log` | Same bug, so it is not caused by the zero-fill shim |

### Analysis (no device)
```bash
python3 compare_tokens.py m70_hf.json m70z_baseline.json           # vs HF, and baseline vs baseline2 / SpD
for m in ngram suffix; do
  python3 hazard_analyze.py m70zspd1_$m.json.probe.jsonl m70zspd1_$m.json m70zspd1_baseline.json
done
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PY margin_check.py m70_hf.json m70pad1_ngram.json
# -> prompt 1 div@17 margin=0.0092, prompt 6 div@27 margin=0.0039 (re-verified when archiving)
```
The check that the accepted counts `hazard_analyze.py` reconstructs agree with the logged
`num_accepted_tokens` at mns=8 (0 mismatches) was an inline snippet and was not saved as a script.

## Output file conventions
- `<tag>_<method>.json`: prompts, generated token ids and text, and SpD counter deltas (drafts and accepted).
- `<tag>_<method>.json.probe.jsonl`: one record per GDN metadata build:
  - number of spec decodes, decodes and prefills;
  - per-row query lengths and draft counts;
  - `num_accepted_tokens`;
  - per-request computed-token state.

  There is one record per KV group, so `hazard_analyze.py` clusters them.
- `<tag>_<method>.log`: the full child stdout and stderr, including QAIC/Triton diagnostics.
- `<tag>_summary.json`: the driver's per-method status and counters.
- Not committed: `ld_*.pt` activation dumps, `__pycache__/`, and `qaic-dumps/` (empty runtime logs).

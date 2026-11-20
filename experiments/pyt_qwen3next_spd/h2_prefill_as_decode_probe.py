# H2 probe: are vLLM 0.30.0 GDN *prefill* kernels correct on QAIC when used for
# 1-token sequences with an initial state?  Upstream does exactly this for the
# non-spec decode rows of a batch that also contains spec-decode rows
# (gdn_attn.py "Reclassify non-spec decodes as prefills when spec decodes exist").
#
# For B in {1,2,4,8} (1 token per request, non-contiguous state slots):
#   (a)  "prefill-as-decode": causal_conv1d_fn(has_initial_state, cache_indices,
#        query_start_loc=arange(B+1), metadata=compute_causal_conv1d_metadata(..))
#        -> fused_post_conv_prep -> chunk_gated_delta_rule (FLA, forward_native)
#        with initial_state=ssm[idx], cu_seqlens, chunk_indices/offsets
#        -> ssm[idx] = final   (mirrors _forward_core num_prefills>0 branch)
#   (b)  decode: causal_conv1d_update(conv_state_indices=idx, validate_data=True)
#        -> fused_sigmoid_gating_delta_rule_update(1D ssm_state_indices)
#        (mirrors _forward_core "elif num_decodes > 0" branch)
#   (b') packed decode: fused_recurrent_gated_delta_rule_packed_decode
#        (_forward_core_decode_non_spec; what a pure non-spec batch uses)
#   (c)  CPU fp32 reference (explicit recurrence).
# Plus one mixed-length case lens=[1,1,3,1] for (a) vs (c).
#
# Usage: python -u h2_prefill_as_decode_probe.py
import os
import sys
import time
import traceback
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_qaic  # noqa: F401,E402

# --- experiment shims (must be applied before any kernel compiles) ----------
from run_qwen3next_spd import (  # noqa: E402
    _shim_accelerator_empty_cache,
    _shim_tl_load_zero,
)

_shim_accelerator_empty_cache()
if os.environ.get("H2_NO_TL_ZERO") != "1":
    _shim_tl_load_zero()

from vllm.platforms import current_platform  # noqa: F401,E402  (qaic plugin)
import wy_fast_zero_pad  # noqa: F401,E402  (recompute_w_u zero-pad patch)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE  # noqa: E402
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata  # noqa: E402

dev = torch.device("qaic:0")

# theo77186/Qwen3-Next-70M-TinyStories linear-attention shapes
H, HV, K, V, W = 1, 2, 128, 128, 4
KD, VD = H * K, HV * V
D = 2 * KD + VD
NSPEC = 3
S = W - 1 + NSPEC  # conv-state length when spec decoding is enabled
POOL = 12  # slot 0 = NULL_BLOCK_ID, never used
DT = torch.float16
SSM_DT = {"fp16": torch.float16, "fp32": torch.float32}[os.environ.get("SSM_DTYPE", "fp16")]
SCALE = K ** -0.5
SLOTS = [5, 2, 9, 11, 3, 7, 1, 10]
ACT = "silu"
print(f"H={H} HV={HV} K={K} V={V} W={W} D={D} S={S} POOL={POOL} dtype={DT} "
      f"ssm_dtype={SSM_DT} FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}", flush=True)

torch.manual_seed(0)
conv_w = (torch.randn(D, W) * 0.5).to(DT)
A_log = torch.log(torch.rand(HV) * 4 + 0.5).float()
dt_bias = (torch.randn(HV) * 0.5).to(DT)


def to_dev(*ts):
    return [t.to(dev) for t in ts]


conv_w_d, A_log_d, dt_bias_d = to_dev(conv_w, A_log, dt_bias)


def sync():
    torch.qaic.synchronize()


RESULTS = []  # (case, comparison, tensor, max_abs, ref_max, rel)


def cmp(case, comp, name, got, ref, tol=1e-2):
    got = got.detach().float().cpu()
    ref = ref.detach().float().cpu()
    err = (got - ref).abs().max().item() if got.numel() else 0.0
    mag = ref.abs().max().item() if ref.numel() else 0.0
    rel = err / max(mag, 1e-12)
    ok = rel < tol and err == err
    RESULTS.append((case, comp, name, err, mag, rel, ok))
    print(f"  [{'OK' if ok else 'MISMATCH'}] {case} {comp} {name}: max_abs={err:.3e} "
          f"ref_max={mag:.3e} rel={rel:.3e}", flush=True)
    return ok


# ---------------------------------------------------------------- reference
def l2n(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def ref_conv(window, x):
    seq = torch.cat([window.float(), x.float()], 0)
    T = x.shape[0]
    out = torch.stack([(seq[t:t + W] * conv_w.float().T).sum(0) for t in range(T)])
    return F.silu(out), seq


def split_qkv(m):
    T = m.shape[0]
    q, k, v = torch.split(m, [KD, KD, VD], dim=-1)
    return q.reshape(T, H, K), k.reshape(T, H, K), v.reshape(T, HV, V)


def ref_gdn(h, q, k, v, a, b):
    h = h.float().clone()
    outs = []
    for t in range(q.shape[0]):
        o_t = torch.empty(HV, V)
        for hv in range(HV):
            ih = hv // (HV // H)
            qq, kk = l2n(q[t, ih].float()) * SCALE, l2n(k[t, ih].float())
            x = a[t, hv].float() + dt_bias[hv].float()
            sp = F.softplus(x) if x <= 20 else x
            gg = -torch.exp(A_log[hv]) * sp
            bb = torch.sigmoid(b[t, hv].float())
            h[hv] *= torch.exp(gg)
            vv = (v[t, hv].float() - h[hv] @ kk) * bb
            h[hv] += torch.outer(vv, kk)
            o_t[hv] = h[hv] @ qq
        outs.append(o_t)
    return torch.stack(outs), h


def reference(conv_sd, ssm, mixed, a, b, idx, lens):
    """Returns per-token output (T,HV,V), and expected conv/ssm pools."""
    new_conv, new_ssm = conv_sd.float().clone(), ssm.float().clone()
    outs = []
    s = 0
    for r, L in enumerate(lens):
        sl = idx[r]
        rco, seq = ref_conv(conv_sd[sl, :W - 1], mixed[s:s + L])
        q, k, v = split_qkv(rco)
        o, h = ref_gdn(ssm[sl], q, k, v, a[s:s + L], b[s:s + L])
        outs.append(o)
        new_conv[sl, :W - 1] = seq[-(W - 1):]
        # positions [W-1:S] of the slot: semantics differ by kernel (see report)
        new_ssm[sl] = h
        s += L
    return torch.cat(outs), new_conv, new_ssm


# ---------------------------------------------------------------- device paths
def path_prefill(conv_sd_d, ssm_d, mixed, a, b, idx, lens):
    """Mirror of _forward_core with num_prefills>0, spec_sequence_masks set
    (so split_non_spec=False): non-spec tokens -> causal_conv1d_fn ->
    fused_post_conv_prep -> chunk rule; builder metadata as in gdn_attn.build."""
    B = len(lens)
    qsl_cpu = torch.zeros(B + 1, dtype=torch.int32)
    qsl_cpu[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    qsl_d = qsl_cpu.to(dev)
    idx_d = torch.tensor(idx, dtype=torch.int32).to(dev)
    has_init_d = torch.ones(B, dtype=torch.bool).to(dev)  # context_lens > 0
    nums_dict, batch_ptr, tco_ptr = compute_causal_conv1d_metadata(qsl_cpu, device=dev)
    chunk_indices = prepare_chunk_indices(qsl_cpu, FLA_CHUNK_SIZE).to(dev)
    chunk_offsets = prepare_chunk_offsets(qsl_cpu, FLA_CHUNK_SIZE).to(dev)
    meta = SimpleNamespace(nums_dict=nums_dict, batch_ptr=batch_ptr,
                           token_chunk_offset_ptr=tco_ptr)
    conv_state = conv_sd_d.transpose(-1, -2)  # SD layout -> (pool, D, S)
    mixed_d, a_d, b_d = to_dev(mixed, a, b)
    mixed_T = mixed_d.transpose(0, 1)
    conv_out = causal_conv1d_fn(
        mixed_T, conv_w_d, None, activation=ACT, conv_states=conv_state,
        has_initial_state=has_init_d, cache_indices=idx_d,
        query_start_loc=qsl_d, metadata=meta,
    ).transpose(0, 1)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out, a=a_d, b=b_d, A_log=A_log_d, dt_bias=dt_bias_d,
        num_k_heads=H, head_k_dim=K, head_v_dim=V, apply_l2norm=True,
        output_g_exp=False,
    )
    q, k, v, g, beta = [x.unsqueeze(0) for x in (q, k, v, g, beta)]
    initial_state = ssm_d[idx_d]
    initial_state[~has_init_d, ...] = 0
    o, final = fla_chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
        output_final_state=True, cu_seqlens=qsl_d, chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets, use_qk_l2norm_in_kernel=False,
    )
    ssm_d[idx_d] = final.to(ssm_d.dtype)
    return o.squeeze(0), final


def path_decode(conv_sd_d, ssm_d, mixed, a, b, idx, packed):
    B = len(idx)
    idx_d = torch.tensor(idx, dtype=torch.int32).to(dev)
    qsl_d = torch.arange(B + 1, dtype=torch.int32).to(dev)
    conv_state = conv_sd_d.transpose(-1, -2)
    mixed_d, a_d, b_d = to_dev(mixed, a, b)
    conv_out = causal_conv1d_update(
        mixed_d, conv_state, conv_w_d, None, ACT,
        conv_state_indices=idx_d[:B], validate_data=not packed,
    )
    if packed:
        out = torch.zeros(B, 1, HV, V, dtype=DT, device=dev)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=conv_out, a=a_d, b=b_d, A_log=A_log_d, dt_bias=dt_bias_d,
            scale=SCALE, initial_state=ssm_d, out=out,
            ssm_state_indices=idx_d[:B], use_qk_l2norm_in_kernel=True,
        )
        return out.squeeze(1)
    q, k, v = split_qkv(conv_out)
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log_d, a=a_d, b=b_d, dt_bias=dt_bias_d,
        q=q.contiguous().unsqueeze(0), k=k.contiguous().unsqueeze(0),
        v=v.contiguous().unsqueeze(0),
        initial_state=ssm_d, inplace_final_state=True,
        cu_seqlens=qsl_d[:B + 1], ssm_state_indices=idx_d,
        use_qk_l2norm_in_kernel=True,
    )
    return o.squeeze(0)


# ---------------------------------------------------------------- driver
def make_inputs(T, seed):
    g = torch.Generator().manual_seed(seed)
    conv_sd = (torch.randn(POOL, S, D, generator=g) * 0.5).to(DT)
    ssm = (torch.randn(POOL, HV, V, K, generator=g) * 0.1).to(SSM_DT)
    mixed = torch.randn(T, D, generator=g).to(DT)
    a = torch.randn(T, HV, generator=g).to(DT)
    b = torch.randn(T, HV, generator=g).to(DT)
    return conv_sd, ssm, mixed, a, b


def run_path(name, fn, conv_sd, ssm):
    conv_sd_d, ssm_d = to_dev(conv_sd, ssm)
    t = time.time()
    try:
        o = fn(conv_sd_d, ssm_d)
        sync()
    except Exception:
        traceback.print_exc()
        print(f"  {name}: EXCEPTION", flush=True)
        return None
    print(f"  {name} ran ({time.time()-t:.1f}s)", flush=True)
    return o.float().cpu(), conv_sd_d.float().cpu(), ssm_d.float().cpu()


def check_vs(case, comp, got, exp, conv_sd, ssm, idx):
    o, cpool, spool = got
    eo, ecpool, espool = exp
    cmp(case, comp, "out", o, eo)
    cmp(case, comp, "conv[slots,0:W-1]", cpool[idx, :W - 1], ecpool[idx, :W - 1])
    cmp(case, comp, "ssm[slots]", spool[idx], espool[idx])
    for r, sl in enumerate(idx[:2]):
        cmp(case, comp, f"ssm[slot {sl}] (req{r})", spool[sl], espool[sl])
    others = [i for i in range(POOL) if i not in idx]
    cmp(case, comp, "conv[other slots] untouched", cpool[others], conv_sd.float()[others],
        tol=1e-6)
    cmp(case, comp, "ssm[other slots] untouched", spool[others], ssm.float()[others],
        tol=1e-6)


def run_case(lens, seed, do_decode):
    B = len(lens)
    idx = SLOTS[:B]
    case = f"B={B} lens={lens}"
    print(f"-- {case} slots={idx}", flush=True)
    conv_sd, ssm, mixed, a, b = make_inputs(sum(lens), seed)
    ref_o, ref_conv_pool, ref_ssm_pool = reference(conv_sd, ssm, mixed, a, b, idx, lens)
    exp = (ref_o, ref_conv_pool, ref_ssm_pool)
    # negative control: same inputs with zero initial ssm state
    ctrl_o, _, _ = reference(conv_sd, torch.zeros_like(ssm), mixed, a, b, idx, lens)
    print(f"  control |ref(init state) - ref(zero state)| out max = "
          f"{(ref_o - ctrl_o).abs().max():.3e} (ref_max {ref_o.abs().max():.3e})",
          flush=True)

    ga = run_path("(a) prefill-as-decode",
                  lambda c, s: path_prefill(c, s, mixed, a, b, idx, lens)[0],
                  conv_sd, ssm)
    if ga is not None:
        check_vs(case, "(a)vs(c)", ga, exp, conv_sd, ssm, idx)
        cmp(case, "(a)", "conv[slots,W-1:S] untouched", ga[1][idx, W - 1:],
            conv_sd.float()[idx, W - 1:], tol=1e-6)
    if not do_decode:
        return
    for nm, packed in (("(b)", False), ("(b')", True)):
        gb = run_path(f"{nm} decode packed={packed}",
                      lambda c, s: path_decode(c, s, mixed, a, b, idx, packed),
                      conv_sd, ssm)
        if gb is None:
            continue
        check_vs(case, f"{nm}vs(c)", gb, exp, conv_sd, ssm, idx)
        # decode shifts the whole S-length buffer by 1 (non-spec): report [W-1:S]
        d = (gb[1][idx, W - 1:] - conv_sd.float()[idx, W - 1:]).abs().max().item()
        print(f"  info {nm} conv[slots,W-1:S] change vs original: {d:.3e}", flush=True)
        if ga is not None:
            cmp(case, f"(a)vs{nm}", "out", ga[0], gb[0])
            cmp(case, f"(a)vs{nm}", "conv[slots,0:W-1]", ga[1][idx, :W - 1],
                gb[1][idx, :W - 1])
            cmp(case, f"(a)vs{nm}", "ssm[slots]", ga[2][idx], gb[2][idx])


if __name__ == "__main__":
    t0 = time.time()
    for i, B in enumerate((1, 2, 4, 8)):
        run_case([1] * B, seed=100 + i, do_decode=True)
    run_case([1, 1, 3, 1], seed=200, do_decode=False)
    print("\n==== SUMMARY ====")
    print(f"{'case':<22} {'comparison':<10} {'tensor':<30} {'max_abs':>10} "
          f"{'ref_max':>10} {'rel':>10}  ok")
    for case, comp, name, err, mag, rel, ok in RESULTS:
        print(f"{case:<22} {comp:<10} {name:<30} {err:>10.3e} {mag:>10.3e} "
              f"{rel:>10.3e}  {'OK' if ok else 'MISMATCH'}")
    nbad = sum(not r[-1] for r in RESULTS)
    print(f"TOTAL {len(RESULTS)} checks, {nbad} mismatches ({time.time()-t0:.1f}s)")

# Probe vLLM's Gated DeltaNet (Qwen3-Next) Triton kernels on QAIC, including
# the speculative-decoding per-position state-slot branches.
#
# Usage: python -u gdn_kernel_probe.py <stage> [tiny]
#   stages: conv_fn | chunk | decode | spec | chain | hazard
#   "tiny" shrinks heads/dims (K=V=32) for a fallback compile check.
#
# Every kernel is called exactly as
#   vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py::_forward_core
#   (and ::_forward_core_decode_non_spec for the packed decode path)
# calls it, on vLLM 0.30.0. Results are compared against an explicit float32
# CPU loop reference written from the kernel source (see ref_* below).
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import torch_qaic  # noqa: F401

from vllm.platforms import current_platform  # noqa: F401  (loads qaic plugin)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

import os as _os
if _os.environ.get("PROBE_ZERO_PAD"):
    import wy_fast_zero_pad  # noqa: F401  (probe-only kernel patch)

dev = torch.device("qaic:0")
STAGE = sys.argv[1]
TINY = len(sys.argv) > 2 and sys.argv[2] == "tiny"

# theo77186/Qwen3-Next-70M-TinyStories linear-attention shapes
H, HV, K, V, W = 1, 2, 128, 128, 4
if TINY:
    H, HV, K, V, W = 1, 2, 32, 32, 4
KD, VD = H * K, HV * V
D = 2 * KD + VD  # conv dim (512 for the real model)
NSPEC = 3  # k drafts -> k+1 = 4 state slots per request
S = W - 1 + NSPEC  # conv-state length with spec decoding
POOL = 10  # state pool size; slot 0 is NULL_BLOCK_ID and is never used
DT = torch.float16
SCALE = K ** -0.5
print(f"stage={STAGE} H={H} HV={HV} K={K} V={V} W={W} D={D} S={S} "
      f"POOL={POOL} dtype={DT} FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}", flush=True)

torch.manual_seed(0)
# parameters (as model params: conv weight/dt_bias model dtype, A_log fp32)
conv_w = (torch.randn(D, W) * 0.5).to(DT)  # conv1d.weight.view(D, W); bias=False
A_log = torch.log(torch.rand(HV) * 4 + 0.5).float()
dt_bias = (torch.randn(HV) * 0.5).to(DT)
ACT = "silu"


def sync():
    torch.qaic.synchronize()


def report(name, got, ref):
    got = got.detach().float().cpu()
    ref = ref.detach().float().cpu()
    err = (got - ref).abs().max().item()
    mag = ref.abs().max().item()
    rel = err / max(mag, 1e-12)
    ok = rel < 1e-2
    print(f"  [{'OK' if ok else 'MISMATCH'}] {name}: max_abs_err={err:.3e} "
          f"ref_max={mag:.3e} rel={rel:.3e}", flush=True)
    return ok


# ---------------------------------------------------------------- references
def ref_conv(window, x, w):
    """window: (W-1, D) history oldest-first; x: (T, D). Returns silu(conv)
    for each x token and the full [window, x] sequence."""
    seq = torch.cat([window, x], 0).float()
    T = x.shape[0]
    out = torch.stack([(seq[t:t + W] * w.float().T).sum(0) for t in range(T)])
    return F.silu(out), seq


def l2n(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def ref_gdn(h, q, k, v, a, b, l2norm=True, g=None, beta=None):
    """Explicit recurrent gated delta rule (matches
    fused_sigmoid_gating_delta_rule_update_kernel math).
    h: (HV, V, K); q,k: (T, H, K); v: (T, HV, V); a,b: (T, HV) raw inputs
    (gating computed here) or g/beta precomputed (log-decay / sigmoid'ed).
    Returns outputs (T, HV, V) and list of states after each token."""
    h = h.float().clone()
    outs, states = [], []
    for t in range(q.shape[0]):
        o_t = torch.empty(HV, V)
        for hv in range(HV):
            ih = hv // (HV // H)
            qq, kk = q[t, ih].float(), k[t, ih].float()
            if l2norm:
                qq, kk = l2n(qq), l2n(kk)
            qq = qq * SCALE
            if g is None:
                x = a[t, hv].float() + dt_bias[hv].float()
                sp = F.softplus(x) if x <= 20 else x
                gg = -torch.exp(A_log[hv]) * sp
                bb = torch.sigmoid(b[t, hv].float())
            else:
                gg, bb = g[t, hv].float(), beta[t, hv].float()
            h[hv] *= torch.exp(gg)
            vv = (v[t, hv].float() - h[hv] @ kk) * bb
            h[hv] += torch.outer(vv, kk)
            o_t[hv] = h[hv] @ qq
        outs.append(o_t)
        states.append(h.clone())
    return torch.stack(outs), states


def split_qkv(mixed):  # mirrors rearrange_mixed_qkv (without leading 1)
    T = mixed.shape[0]
    q, k, v = torch.split(mixed, [KD, KD, VD], dim=-1)
    return (q.reshape(T, H, K).contiguous(), k.reshape(T, H, K).contiguous(),
            v.reshape(T, HV, V).contiguous())


def new_pools():
    # SD layout (default): kv_cache[0] is (pool, S, D); the layer passes
    # conv_state = kv_cache[0].transpose(-1, -2) -> (pool, D, S).
    conv_sd = (torch.randn(POOL, S, D) * 0.5).to(DT)
    ssm = (torch.randn(POOL, HV, V, K) * 0.1).to(DT)
    return conv_sd, ssm


def rand_tokens(T):
    mixed = torch.randn(T, D).to(DT)
    a = torch.randn(T, HV).to(DT)
    b = torch.randn(T, HV).to(DT)
    return mixed, a, b


def to_dev(*ts):
    return [t.to(dev) for t in ts]


# ---------------------------------------------------------- device wrappers
def dev_spec_step(conv_sd_d, ssm_d, mixed, a, b, spec_idx, num_acc):
    """One spec-decode step for all requests, exactly as _forward_core's spec
    branch (num_prefills == num_decodes == 0). Mutates the device pools."""
    nreq, kp1 = spec_idx.shape
    qsl = torch.arange(0, nreq * kp1 + 1, kp1, dtype=torch.int32)
    spec_idx_d, num_acc_d, qsl_d = to_dev(spec_idx, num_acc, qsl)
    conv_state = conv_sd_d.transpose(-1, -2)
    mixed_d, a_d, b_d = to_dev(mixed, a, b)
    conv_out = causal_conv1d_update(
        mixed_d, conv_state, conv_w_d, None, ACT,
        conv_state_indices=spec_idx_d[:, 0][:nreq],
        num_accepted_tokens=num_acc_d,
        query_start_loc=qsl_d,
        max_query_len=spec_idx_d.size(-1),
        validate_data=False,
    )
    q, k, v = split_qkv(conv_out)
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log_d, a=a_d, b=b_d, dt_bias=dt_bias_d,
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        initial_state=ssm_d, inplace_final_state=True,
        cu_seqlens=qsl_d[:nreq + 1],
        ssm_state_indices=spec_idx_d,
        num_accepted_tokens=num_acc_d,
        use_qk_l2norm_in_kernel=True,
    )
    return conv_out, o.squeeze(0)


def dev_nonspec_decode(conv_sd_d, ssm_d, mixed, a, b, idx, packed):
    """Non-spec decode (1 token/request), 1D state indices.
    packed=True  -> _forward_core_decode_non_spec (default when the whole batch
                    is non-spec decode: VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1)
    packed=False -> _forward_core decode branch (used when spec requests are
                    present in the same batch)."""
    n = idx.shape[0]
    idx_d, = to_dev(idx)
    conv_state = conv_sd_d.transpose(-1, -2)
    mixed_d, a_d, b_d = to_dev(mixed, a, b)
    conv_out = causal_conv1d_update(
        mixed_d, conv_state, conv_w_d, None, ACT,
        conv_state_indices=idx_d[:n], validate_data=not packed,
    )
    if packed:
        out = torch.zeros(n, HV, V, dtype=DT, device=dev)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=conv_out, a=a_d, b=b_d, A_log=A_log_d, dt_bias=dt_bias_d,
            scale=SCALE, initial_state=ssm_d, out=out[:n].unsqueeze(1),
            ssm_state_indices=idx_d[:n], use_qk_l2norm_in_kernel=True,
        )
        return conv_out, out
    q, k, v = split_qkv(conv_out)
    qsl_d = torch.arange(0, n + 1, dtype=torch.int32).to(dev)
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log_d, a=a_d, b=b_d, dt_bias=dt_bias_d,
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        initial_state=ssm_d, inplace_final_state=True,
        cu_seqlens=qsl_d[:n + 1], ssm_state_indices=idx_d,
        use_qk_l2norm_in_kernel=True,
    )
    return conv_out, o.squeeze(0)


def dev_prefill_1tok(conv_sd_d, ssm_d, mixed, a, b, idx):
    """1-token non-spec request in a batch that also has spec requests:
    gdn_attn.py reclassifies it as a prefill (has_initial_state=True), so it
    goes through causal_conv1d_fn + fused_post_conv_prep + chunk rule with
    state index block_table[:,0]."""
    n = idx.shape[0]
    cu_cpu = torch.arange(0, n + 1, dtype=torch.int32)
    idx_d, cu_d = to_dev(idx, cu_cpu)
    hi_d = torch.ones(n, dtype=torch.bool).to(dev)
    conv_state = conv_sd_d.transpose(-1, -2)
    mixed_d, a_d, b_d = to_dev(mixed, a, b)
    conv_out = causal_conv1d_fn(
        mixed_d.transpose(0, 1), conv_w_d, None, activation=ACT,
        conv_states=conv_state, has_initial_state=hi_d, cache_indices=idx_d,
        query_start_loc=cu_d, metadata=None,
    ).transpose(0, 1)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out.contiguous(), a=a_d, b=b_d, A_log=A_log_d,
        dt_bias=dt_bias_d, num_k_heads=H, head_k_dim=K, head_v_dim=V,
        apply_l2norm=True, output_g_exp=False,
    )
    init = ssm_d[idx_d]
    o, final = fla_chunk_gated_delta_rule(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        g=g.unsqueeze(0), beta=beta.unsqueeze(0), initial_state=init,
        output_final_state=True, cu_seqlens=cu_d,
        chunk_indices=prepare_chunk_indices(cu_cpu, FLA_CHUNK_SIZE).to(dev),
        chunk_offsets=prepare_chunk_offsets(cu_cpu, FLA_CHUNK_SIZE).to(dev),
        use_qk_l2norm_in_kernel=False,
    )
    ssm_d[idx_d] = final.to(ssm_d.dtype)
    return conv_out, o.squeeze(0)


conv_w_d, A_log_d, dt_bias_d = to_dev(conv_w, A_log, dt_bias)


# -------------------------------------------------------------------- stages
def stage_conv_fn():
    lens = [40, 24]
    cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32)
    cache_idx = torch.tensor([5, 3], dtype=torch.int32)
    has_init = torch.tensor([True, False])
    conv_sd, _ = new_pools()
    mixed, _, _ = rand_tokens(sum(lens))
    conv_sd_d, mixed_d, cu_d, ci_d, hi_d = to_dev(conv_sd, mixed, cu, cache_idx,
                                                  has_init)
    conv_state = conv_sd_d.transpose(-1, -2)
    t = time.time()
    out = causal_conv1d_fn(
        mixed_d.transpose(0, 1), conv_w_d, None, activation=ACT,
        conv_states=conv_state, has_initial_state=hi_d, cache_indices=ci_d,
        query_start_loc=cu_d, metadata=None,
    ).transpose(0, 1)
    sync()
    print(f"causal_conv1d_fn ran ({time.time()-t:.1f}s)", flush=True)
    ok = True
    new_sd = conv_sd_d.cpu()
    for r in range(2):
        x = mixed[cu[r]:cu[r + 1]]
        win = conv_sd[cache_idx[r], :W - 1] if has_init[r] else torch.zeros(W - 1, D)
        ref_o, seq = ref_conv(win, x, conv_w)
        ok &= report(f"req{r} conv out", out[cu[r]:cu[r + 1]], ref_o)
        ok &= report(f"req{r} conv_state[0:W-1] (last W-1 tokens)",
                     new_sd[cache_idx[r], :W - 1], seq[-(W - 1):])
        ok &= report(f"req{r} conv_state[W-1:S] untouched",
                     new_sd[cache_idx[r], W - 1:], conv_sd[cache_idx[r], W - 1:])
    untouched = [i for i in range(POOL) if i not in cache_idx.tolist()]
    ok &= report("other slots untouched", new_sd[untouched], conv_sd[untouched])
    return ok


def stage_chunk():
    lens = [40, 24] if not TINY else [40, 24]
    if len(sys.argv) > 3:
        lens = [int(x) for x in sys.argv[3].split(",")]
    N = len(lens)
    T = sum(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    # post-conv activations (what causal_conv1d_fn produced)
    conv_out = (torch.randn(T, D)).to(DT)
    _, a, b = rand_tokens(T)
    _, ssm = new_pools()
    state_idx = torch.tensor([5, 3, 7, 2][:N], dtype=torch.int32)
    has_init = torch.tensor([True, False, True, False][:N])
    conv_out_d, a_d, b_d, ssm_d, cu_d = to_dev(conv_out, a, b, ssm, cu)

    t = time.time()
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out_d, a=a_d, b=b_d, A_log=A_log_d, dt_bias=dt_bias_d,
        num_k_heads=H, head_k_dim=K, head_v_dim=V, apply_l2norm=True,
        output_g_exp=False,
    )
    sync()
    print(f"fused_post_conv_prep ran ({time.time()-t:.1f}s)", flush=True)
    rq, rk, rv = split_qkv(conv_out.float())
    xg = a.float() + dt_bias.float()
    rg = -torch.exp(A_log) * torch.where(xg <= 20, F.softplus(xg), xg)
    ok = True
    ok &= report("post_conv q (l2norm)", q, l2n(rq))
    ok &= report("post_conv k (l2norm)", k, l2n(rk))
    ok &= report("post_conv v", v, rv)
    ok &= report("post_conv g", g, rg)
    ok &= report("post_conv beta", beta, torch.sigmoid(b.float()))

    # ChunkGatedDeltaRule.forward_native (QAIC is not CUDA -> "triton" backend)
    ssm_idx_d = state_idx.to(dev)
    initial_state = ssm_d[ssm_idx_d]
    initial_state[~has_init.to(dev), ...] = 0
    ci = prepare_chunk_indices(cu, FLA_CHUNK_SIZE)
    co = prepare_chunk_offsets(cu, FLA_CHUNK_SIZE)
    t = time.time()
    o, final = fla_chunk_gated_delta_rule(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        g=g.unsqueeze(0), beta=beta.unsqueeze(0),
        initial_state=initial_state, output_final_state=True,
        cu_seqlens=cu_d, chunk_indices=ci.to(dev), chunk_offsets=co.to(dev),
        use_qk_l2norm_in_kernel=False,
    )
    sync()
    print(f"chunk_gated_delta_rule ran ({time.time()-t:.1f}s)", flush=True)
    ssm_d[ssm_idx_d] = final.to(ssm_d.dtype)
    sync()
    o = o.squeeze(0).cpu()
    for r in range(N):
        s, e = cu[r].item(), cu[r + 1].item()
        h0 = ssm[state_idx[r]].float() if has_init[r] else torch.zeros(HV, V, K)
        ro, st = ref_gdn(h0, rq[s:e], rk[s:e], rv[s:e], a[s:e], b[s:e])
        ok &= report(f"req{r} chunk out (len {e-s}, init={bool(has_init[r])})",
                     o[s:e], ro)
        ok &= report(f"req{r} final_state", final[r], st[-1])
        ok &= report(f"req{r} ssm pool slot {state_idx[r].item()} written",
                     ssm_d[state_idx[r]], st[-1])
    return ok


def stage_decode():
    ok = True
    idx = torch.tensor([5, 3], dtype=torch.int32)
    for packed in (True, False):
        print(f"-- non-spec decode, packed={packed} "
              f"({'fused_recurrent_..._packed_decode' if packed else 'fused_sigmoid_gating (1D idx)'})",
              flush=True)
        conv_sd, ssm = new_pools()
        mixed, a, b = rand_tokens(2)
        conv_sd_d, ssm_d = to_dev(conv_sd, ssm)
        t = time.time()
        conv_out, o = dev_nonspec_decode(conv_sd_d, ssm_d, mixed, a, b, idx, packed)
        sync()
        print(f"   ran ({time.time()-t:.1f}s)", flush=True)
        new_sd, new_ssm = conv_sd_d.cpu(), ssm_d.cpu()
        for r in range(2):
            sl = idx[r].item()
            rco, seq = ref_conv(conv_sd[sl, :W - 1], mixed[r:r + 1], conv_w)
            ok &= report(f"req{r} conv_update out", conv_out[r:r + 1], rco)
            ok &= report(f"req{r} conv_state[0:W-1] shifted", new_sd[sl, :W - 1],
                         seq[1:])
            rq, rk, rv = split_qkv(rco)
            ro, st = ref_gdn(ssm[sl], rq, rk, rv, a[r:r + 1], b[r:r + 1])
            ok &= report(f"req{r} recurrent out", o[r], ro[0])
            ok &= report(f"req{r} ssm slot {sl} final", new_ssm[sl], st[-1])
    return ok


SPEC_IDX = torch.tensor([[5, 2, 7, 1], [3, 6, 8, 4]], dtype=torch.int32)


def ref_spec_req(conv_sd, ssm, n, acc, mixed, a, b):
    """Reference for one request of a spec step, derived from the kernels'
    IS_SPEC_DECODING semantics: conv window = buf[a-1 : a-1+W-1] of slot
    spec[n,0]; new buf = [old[a : a+W-2], x0..xk]; recurrent initial state =
    slot spec[n, a-1]; slot spec[n,t] <- state after tokens 0..t."""
    sl0 = SPEC_IDX[n, 0].item()
    off = acc - 1
    win = conv_sd[sl0, off:off + W - 1]
    rco, _ = ref_conv(win, mixed, conv_w)
    new_buf = torch.cat([conv_sd[sl0, off + 1:off + W - 1].float(), mixed.float()], 0)
    rq, rk, rv = split_qkv(rco)
    h0 = ssm[SPEC_IDX[n, acc - 1].item()]
    ro, st = ref_gdn(h0, rq, rk, rv, a, b)
    return rco, new_buf, ro, st


def stage_spec():
    ok = True
    kp1 = NSPEC + 1
    for acc in (1, 2, 3, 4):
        accs = [acc, (acc % 4) + 1]
        num_acc = torch.tensor(accs, dtype=torch.int32)
        print(f"-- spec step: num_accepted_tokens={accs}", flush=True)
        conv_sd, ssm = new_pools()
        mixed, a, b = rand_tokens(2 * kp1)
        conv_sd_d, ssm_d = to_dev(conv_sd, ssm)
        t = time.time()
        conv_out, o = dev_spec_step(conv_sd_d, ssm_d, mixed, a, b, SPEC_IDX, num_acc)
        sync()
        print(f"   ran ({time.time()-t:.1f}s)", flush=True)
        new_sd, new_ssm, conv_out, o = conv_sd_d.cpu(), ssm_d.cpu(), conv_out.cpu(), o.cpu()
        for n in range(2):
            s, e = n * kp1, (n + 1) * kp1
            rco, new_buf, ro, st = ref_spec_req(conv_sd, ssm, n, accs[n],
                                                mixed[s:e], a[s:e], b[s:e])
            sl0 = SPEC_IDX[n, 0].item()
            ok &= report(f"4a req{n} a={accs[n]} conv out", conv_out[s:e], rco)
            ok &= report(f"4a req{n} conv buf slot {sl0} (shift a-1={accs[n]-1})",
                         new_sd[sl0], new_buf)
            ok &= report(f"4b req{n} recurrent out (init from slot "
                         f"{SPEC_IDX[n, accs[n]-1].item()})", o[s:e], ro)
            for tt in range(kp1):
                slt = SPEC_IDX[n, tt].item()
                ok &= report(f"4b req{n} slot[{n},{tt}]={slt} == state after tok 0..{tt}",
                             new_ssm[slt], st[tt])
            # negative control: initial state from slot [n,0] would differ
            if accs[n] > 1:
                ro_wrong, _ = ref_gdn(ssm[SPEC_IDX[n, 0].item()],
                                      *split_qkv(rco), a[s:e], b[s:e])
                d = (o[s:e].float() - ro_wrong).abs().max().item()
                print(f"     control: |out - ref_from_slot[n,0]|max = {d:.3e} "
                      f"(should be >> err)", flush=True)
        used = SPEC_IDX.flatten().tolist()
        others = [i for i in range(POOL) if i not in used]
        ok &= report("unused ssm slots untouched", new_ssm[others], ssm[others])
        conv_other = [i for i in range(POOL) if i not in SPEC_IDX[:, 0].tolist()]
        ok &= report("conv slots other than spec[:,0] untouched",
                     new_sd[conv_other], conv_sd[conv_other])
    return ok


class RefStream:
    """Sequential fp32 reference of one request's accepted token stream."""

    def __init__(self, conv_window, h):
        self.win = conv_window.float().clone()  # (W-1, D)
        self.h = h.float().clone()

    def run(self, mixed, a, b, commit):
        rco, seq = ref_conv(self.win, mixed, conv_w)
        rq, rk, rv = split_qkv(rco)
        ro, st = ref_gdn(self.h, rq, rk, rv, a, b)
        if commit > 0:
            self.win = seq[commit:commit + W - 1].clone()
            self.h = st[commit - 1].clone()
        return ro, st


def stage_chain():
    ok = True
    kp1 = NSPEC + 1
    for a1 in (1, 2, 3, 4):
        accs1 = [a1, (a1 % 4) + 1]
        print(f"-- chain: step1 (num_accepted=[1,1]) then step2 "
              f"num_accepted={accs1}", flush=True)
        conv_sd, ssm = new_pools()
        conv_sd_d, ssm_d = to_dev(conv_sd, ssm)
        refs = [RefStream(conv_sd[SPEC_IDX[n, 0], :W - 1], ssm[SPEC_IDX[n, 0]])
                for n in range(2)]
        # step 1: previous step accepted 1 (e.g. first decode after prefill)
        m1, a_1, b_1 = rand_tokens(2 * kp1)
        _, o1 = dev_spec_step(conv_sd_d, ssm_d, m1, a_1, b_1, SPEC_IDX,
                              torch.tensor([1, 1], dtype=torch.int32))
        for n in range(2):
            s, e = n * kp1, (n + 1) * kp1
            ro, _ = refs[n].run(m1[s:e], a_1[s:e], b_1[s:e], commit=accs1[n])
            ok &= report(f"step1 req{n} out", o1[s:e], ro)
        # step 2 reads slot [n, a1-1] and conv offset a1-1
        m2, a_2, b_2 = rand_tokens(2 * kp1)
        _, o2 = dev_spec_step(conv_sd_d, ssm_d, m2, a_2, b_2, SPEC_IDX,
                              torch.tensor(accs1, dtype=torch.int32))
        sync()
        new_ssm = ssm_d.cpu()
        for n in range(2):
            s, e = n * kp1, (n + 1) * kp1
            ro, st = refs[n].run(m2[s:e], a_2[s:e], b_2[s:e], commit=0)
            ok &= report(f"step2 req{n} (a1={accs1[n]}) out vs accepted-prefix ref",
                         o2[s:e], ro)
            for tt in range(kp1):
                ok &= report(f"step2 req{n} slot[{n},{tt}] final state",
                             new_ssm[SPEC_IDX[n, tt]], st[tt])
    return ok


def stage_hazard():
    """Spec step where req0 accepts a=3, followed by a NON-spec decode of req0
    using vLLM's non-spec indexing (block_table[:,0] -> slot SPEC_IDX[0,0],
    num_accepted_tokens=None)."""
    ok = True
    kp1 = NSPEC + 1
    acc = 3
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["packed", "fsg", "prefill1"]
    for mode in modes:
        packed = mode
        print(f"-- hazard: spec step (prev a=[1,1]) -> req0 accepts {acc} -> "
              f"non-spec path mode={mode}", flush=True)
        torch.manual_seed(1)
        conv_sd, ssm = new_pools()
        conv_sd_d, ssm_d = to_dev(conv_sd, ssm)
        sl0 = SPEC_IDX[0, 0].item()
        good = RefStream(conv_sd[sl0, :W - 1], ssm[sl0])
        stale = RefStream(conv_sd[sl0, :W - 1], ssm[sl0])
        m1, a_1, b_1 = rand_tokens(2 * kp1)
        _, o1 = dev_spec_step(conv_sd_d, ssm_d, m1, a_1, b_1, SPEC_IDX,
                              torch.tensor([1, 1], dtype=torch.int32))
        good.run(m1[:kp1], a_1[:kp1], b_1[:kp1], commit=acc)
        stale.run(m1[:kp1], a_1[:kp1], b_1[:kp1], commit=1)
        sync()
        ssm_after = ssm_d.cpu()
        conv_after = conv_sd_d.cpu()
        print(f"   slot {sl0} (block_table[0,0]) holds: "
              f"|ssm - state_after_accepted_prefix|max="
              f"{(ssm_after[sl0].float() - good.h).abs().max():.3e}, "
              f"|ssm - state_after_tok0|max="
              f"{(ssm_after[sl0].float() - stale.h).abs().max():.3e}", flush=True)
        print(f"   conv buf slot {sl0}[0:W-1] vs correct window: "
              f"{(conv_after[sl0, :W-1].float() - good.win).abs().max():.3e}, "
              f"vs window-ending-at-tok0: "
              f"{(conv_after[sl0, :W-1].float() - stale.win).abs().max():.3e}",
              flush=True)
        # non-spec decode of req0 only (req1 kept on its spec slot 0 too)
        m2, a_2, b_2 = rand_tokens(1)
        idx = torch.tensor([sl0], dtype=torch.int32)
        if mode == "prefill1":
            _, o2 = dev_prefill_1tok(conv_sd_d, ssm_d, m2, a_2, b_2, idx)
        else:
            _, o2 = dev_nonspec_decode(conv_sd_d, ssm_d, m2, a_2, b_2, idx,
                                       mode == "packed")
        sync()
        ro_good, _ = good.run(m2, a_2, b_2, commit=0)
        ro_stale, _ = stale.run(m2, a_2, b_2, commit=0)
        e_good = (o2.cpu().float() - ro_good[0]).abs().max().item()
        e_stale = (o2.cpu().float() - ro_stale[0]).abs().max().item()
        mag = ro_good.abs().max().item()
        print(f"   non-spec out vs CORRECT ref (state after {acc} accepted): "
              f"max_abs_err={e_good:.3e} (ref_max={mag:.3e})", flush=True)
        print(f"   non-spec out vs STALE ref (state after tok0 only): "
              f"max_abs_err={e_stale:.3e}", flush=True)
        import math
        if math.isnan(e_good) or math.isnan(e_stale):
            verdict = "INCONCLUSIVE (kernel output NaN)"
        elif e_stale < e_good:
            verdict = "STALE (reads slot [0,0] = post-token-0 state)"
        else:
            verdict = "CORRECT"
        print(f"   HAZARD VERDICT (mode={mode}): {verdict}", flush=True)
        ok &= e_stale < 1e-2 * max(mag, 1e-12)  # kernel itself behaved as indexed
    return ok


def stage_chunk_parts():
    """Bisect chunk_gated_delta_rule_fwd sub-kernels (same inputs as 'chunk')."""
    from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h)
    from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o
    from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd)
    from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
    from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
    from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd
    lens = [40, 24]
    if len(sys.argv) > 3:
        lens = [int(x) for x in sys.argv[3].split(",")]
    N, T = len(lens), sum(lens)
    BT = FLA_CHUNK_SIZE
    assert all(L <= BT for L in lens), "bisect refs assume 1 chunk/seq"
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    conv_out = torch.randn(T, D).to(DT)
    _, a, b = rand_tokens(T)
    _, ssm = new_pools()
    init = ssm[[5, 3][:N]].clone()
    init[1:] = 0
    conv_out_d, a_d, b_d, cu_d, init_d = to_dev(conv_out, a, b, cu, init)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out_d, a=a_d, b=b_d, A_log=A_log_d, dt_bias=dt_bias_d,
        num_k_heads=H, head_k_dim=K, head_v_dim=V, apply_l2norm=True)
    q, k, v, g, beta = [x.unsqueeze(0) for x in (q, k, v, g, beta)]
    ci = prepare_chunk_indices(cu, BT).to(dev)
    co = prepare_chunk_offsets(cu, BT).to(dev)

    def nan(name, x):
        x = x.float().cpu()
        print(f"  {name}: shape={tuple(x.shape)} nan={torch.isnan(x).sum().item()} "
              f"inf={torch.isinf(x).sum().item()} absmax="
              f"{torch.nan_to_num(x).abs().max().item():.3e}", flush=True)

    def run(name, fn):
        t = time.time()
        r = fn()
        sync()
        print(f"{name} ran ({time.time()-t:.1f}s)", flush=True)
        return r

    kc, vc, bc = k[0].float().cpu(), v[0].float().cpu(), beta[0].float().cpu()
    gc = run("chunk_local_cumsum", lambda: chunk_local_cumsum(
        g, chunk_size=BT, cu_seqlens=cu_d, chunk_indices=ci))
    nan("g_cumsum", gc)
    g0 = g[0].float().cpu()
    ref_gc = torch.cat([g0[cu[r]:cu[r + 1]].cumsum(0) for r in range(N)])
    report("g_cumsum", gc[0], ref_gc)
    gcc = gc[0].float().cpu()
    A = run("chunk_scaled_dot_kkt_fwd", lambda: chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=gc, cu_seqlens=cu_d, chunk_indices=ci,
        output_dtype=torch.float32))
    nan("A", A)
    # ref A[t, hv, j] (row t, col j within chunk), strictly lower
    refA = torch.zeros(T, HV, BT)
    for r in range(N):
        s, e = cu[r].item(), cu[r + 1].item()
        for hv in range(HV):
            kk = kc[s:e, hv // (HV // H)]
            M = (bc[s:e, hv, None] * kk) @ kk.T * torch.exp(
                gcc[s:e, hv, None] - gcc[None, s:e, hv])
            refA[s:e, hv, :e - s] = torch.tril(M, -1)
    report("A (beta*kk^T*exp(dg), strict lower)", A[0], refA)
    Ac = A[0].float().cpu()
    Ai = run("solve_tril", lambda: solve_tril(
        A=A, cu_seqlens=cu_d, chunk_indices=ci, output_dtype=k.dtype))
    nan("Ai", Ai)
    refAi = torch.zeros(T, HV, BT)
    for r in range(N):
        s, e = cu[r].item(), cu[r + 1].item()
        for hv in range(HV):
            L = e - s
            refAi[s:e, hv, :L] = torch.linalg.inv(torch.eye(L) + Ac[s:e, hv, :L])
    report("Ai = (I+A)^-1", Ai[0], refAi)
    Aic = Ai[0].float().cpu()
    w, u = run("recompute_w_u_fwd", lambda: recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=Ai, g_cumsum=gc, cu_seqlens=cu_d, chunk_indices=ci))
    nan("w", w)
    wn = torch.isnan(w[0].float().cpu())
    for r in range(N):
        s_, e_ = cu[r].item(), cu[r + 1].item()
        for hv in range(HV):
            blk = wn[s_:e_, hv]
            rows = blk.any(-1).nonzero().flatten().tolist()
            print(f"    w NaN req{r} hv{hv}: {blk.sum().item()} elems, rows(local)="
                  f"{rows[:8]}{'...' if len(rows) > 8 else ''} n_rows={len(rows)} "
                  f"cols_per_row={blk.sum(-1)[rows].unique().tolist() if rows else []}",
                  flush=True)
    nan("u", u)
    refw, refu = torch.zeros(T, HV, K), torch.zeros(T, HV, V)
    for r in range(N):
        s, e = cu[r].item(), cu[r + 1].item()
        L = e - s
        for hv in range(HV):
            M = Aic[s:e, hv, :L]
            refu[s:e, hv] = M @ (vc[s:e, hv] * bc[s:e, hv, None])
            refw[s:e, hv] = M @ (kc[s:e, hv // (HV // H)] * bc[s:e, hv, None]
                                 * torch.exp(gcc[s:e, hv, None]))
    report("w", w[0], refw)
    report("u", u[0], refu)
    h, v_new, final = run("chunk_gated_delta_rule_fwd_h", lambda:
                          chunk_gated_delta_rule_fwd_h(
                              k=k, w=w, u=u, g=gc, initial_state=init_d,
                              output_final_state=True, cu_seqlens=cu_d,
                              chunk_indices=ci, chunk_offsets=co))
    nan("h (per-chunk initial states)", h)
    nan("v_new", v_new)
    nan("final_state", final)
    for r in range(N):  # one chunk per request -> h[0, r] is req r's start state
        report(f"h[0,{r}] == initial state req{r}", h[0, r], init[r].float())
    o = run("chunk_fwd_o", lambda: chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=gc, scale=SCALE, cu_seqlens=cu_d,
        chunk_indices=ci))
    nan("o", o)
    rq, rk, rv = split_qkv(conv_out.float())
    for r in range(N):
        s, e = cu[r].item(), cu[r + 1].item()
        ro, st = ref_gdn(init[r], rq[s:e], rk[s:e], rv[s:e], a[s:e], b[s:e])
        report(f"req{r} final_state", final[r], st[-1])
        report(f"req{r} o", o[0, s:e], ro)
    ok = not torch.isnan(o.float().cpu()).any().item()
    return ok


STAGES = {"chunk_parts": stage_chunk_parts, "conv_fn": stage_conv_fn, "chunk": stage_chunk, "decode": stage_decode,
          "spec": stage_spec, "chain": stage_chain, "hazard": stage_hazard}

if __name__ == "__main__":
    t0 = time.time()
    try:
        res = STAGES[STAGE]()
        print(f"STAGE {STAGE} RESULT: {'OK' if res else 'NUMERIC MISMATCH'} "
              f"({time.time()-t0:.1f}s)", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"STAGE {STAGE} RESULT: EXCEPTION ({time.time()-t0:.1f}s)", flush=True)
        sys.exit(1)

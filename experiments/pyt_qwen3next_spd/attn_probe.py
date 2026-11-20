"""Capture layers.3.self_attn.{rotary_emb, attn} in/out in vLLM PYT on QAIC and
recompute each on CPU (upstream RotaryEmbedding.forward_native, causal SDPA w/ GQA)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import layer_diff as L  # noqa: E402
import run_qwen3next_spd as R  # noqa: E402

LAYER = int(os.environ.get("PROBE_LAYER", 3))


def cpu(x):
    return x.detach().float().cpu() if isinstance(x, torch.Tensor) else x


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    R._shim_accelerator_empty_cache()
    R._shim_rmsnorm_gated_hidden_size()
    R._shim_gemma_rmsnorm_residual()
    R._shim_platform_hybrid_kv_cache()
    R._shim_qwen_triton_warmup("skip")
    import wy_fast_zero_pad  # noqa: F401
    ids = R._prompt_ids(L.MODEL)[0]
    T = len(ids)
    llm = LLM(model=L.MODEL, dtype="float16", max_num_seqs=1, max_model_len=512,
              enforce_eager=True, async_scheduling=False, enable_prefix_caching=False,
              kv_cache_memory_bytes=1 << 30)
    cap = {}

    def install(m):
        sa = dict(m.named_modules())[f"model.layers.{LAYER}.self_attn"]
        cap["meta"] = dict(H=sa.num_heads, KV=sa.num_kv_heads, D=sa.head_dim,
                           scale=sa.scaling, rdim=sa.rotary_emb.rotary_dim,
                           neox=sa.rotary_emb.is_neox_style,
                           rope_cls=type(sa.rotary_emb).__name__,
                           attn_impl=type(sa.attn.impl).__name__)
        cap["cos_sin"] = cpu(sa.rotary_emb.cos_sin_cache)

        def rh(mod, args, out):
            if args[0].numel() == T and "rope" not in cap:
                cap["rope"] = ([cpu(a) for a in args], [cpu(o) for o in out])

        def ah(mod, args, out):
            if args[0].shape[0] == T and "attn" not in cap:
                cap["attn"] = ([cpu(a) for a in args[:3]], cpu(out))
        sa.rotary_emb.register_forward_hook(rh)
        sa.attn.register_forward_hook(ah)
    llm.apply_model(install)
    llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=1))
    md = cap["meta"]
    print("meta", {k: v for k, v in md.items()}, flush=True)
    H, KV, D = md["H"], md["KV"], md["D"]

    # --- RoPE reference (upstream forward_native semantics, NeoX, partial rotary) ---
    (pos, q_in, k_in), (q_out, k_out) = cap["rope"]
    pos = pos.long().flatten()
    cs = cap["cos_sin"].index_select(0, pos)
    cos, sin = cs.chunk(2, dim=-1)

    def rot(x, n):
        x = x.view(T, n, D)
        r, p = x[..., :md["rdim"]], x[..., md["rdim"]:]
        x1, x2 = r.chunk(2, dim=-1)
        c, s = cos[:, None, :], sin[:, None, :]
        r = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
        return torch.cat((r, p), dim=-1).reshape(T, n * D)
    for nm, ref, got in (("rope.q", rot(q_in, H), q_out), ("rope.k", rot(k_in, KV), k_out)):
        d = (ref - got).abs().max().item()
        print(f"{nm}: max|ref|={ref.abs().max():.3e} maxabs={d:.3e} rel={d / ref.abs().max():.2e}"
              f"  rot-part rel={((ref - got).view(T, -1, D)[..., :md['rdim']].abs().max() / ref.abs().max()):.2e}"
              f"  pass-part rel={((ref - got).view(T, -1, D)[..., md['rdim']:].abs().max() / ref.abs().max()):.2e}",
              flush=True)

    # --- attention reference: causal SDPA with GQA on the captured q,k,v ---
    (q, k, v), o = cap["attn"]
    qh = q.view(T, H, D).transpose(0, 1)
    kh = k.view(T, KV, D).transpose(0, 1).repeat_interleave(H // KV, 0)
    vh = v.view(T, KV, D).transpose(0, 1).repeat_interleave(H // KV, 0)
    ref = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, is_causal=True, scale=md["scale"]).transpose(0, 1).reshape(T, H * D)
    o = o.reshape(T, -1)
    d = (ref - o).abs().max().item()
    print(f"attn: out shape {tuple(o.shape)} max|ref|={ref.abs().max():.3e} maxabs={d:.3e} "
          f"rel={d / ref.abs().max():.2e}", flush=True)
    per_tok = (ref - o).abs().amax(-1)
    print("attn per-token maxabs (first 8 / last 4):", per_tok[:8].tolist(), per_tok[-4:].tolist())
    # non-causal check to detect missing mask
    nc = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, scale=md["scale"]).transpose(0, 1).reshape(T, H * D)
    print(f"attn vs NON-causal ref rel={(nc - o).abs().max() / nc.abs().max():.2e}")


if __name__ == "__main__":
    main()

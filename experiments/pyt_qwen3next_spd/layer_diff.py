"""Per-sublayer activation diff, vLLM PYT on QAIC (fp16) vs HF CPU fp32, for one
prompt's prefill.

  python layer_diff.py hf    <out.pt> [prompt_idx]
  python layer_diff.py vllm  <out.pt> [prompt_idx]   (same env/shims as run_qwen3next_spd)
  python layer_diff.py cmp   <hf.pt> <vllm.pt>
"""
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import run_qwen3next_spd as R  # noqa: E402

MODEL = "theo77186/Qwen3-Next-70M-TinyStories"
PAT = re.compile(r"(^|\.)(embed_tokens|layers\.\d+\.(input_layernorm|linear_attn|self_attn|"
                 r"post_attention_layernorm|mlp)|norm)$")


def _t(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    return x if isinstance(x, torch.Tensor) else None


def hook_all(model, store, ntok):
    def mk(name):
        def h(mod, args, kwargs, out):
            if name in store:
                return
            inp = _t(args) if args else _t(kwargs.get("hidden_states", kwargs.get("input_ids")))
            o = _t(out)
            if o is None:
                return
            o2 = o.reshape(-1, o.shape[-1])
            if o2.shape[0] != ntok:
                return
            rec = {"out": o2.detach().float().cpu()}
            if inp is not None and inp.is_floating_point():
                rec["in"] = inp.reshape(-1, inp.shape[-1]).detach().float().cpu()
            store[name] = rec
        return h

    n = 0
    for name, mod in model.named_modules():
        if PAT.search(name):
            key = re.sub(r"^(model\.|language_model\.model\.)", "", name)
            mod.register_forward_hook(mk(key), with_kwargs=True)
            n += 1
    print(f"hooked {n} modules", flush=True)


def run_hf(out, idx):
    from transformers import AutoModelForCausalLM
    ids = R._prompt_ids(MODEL)[idx]
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    store = {}
    hook_all(m, store, len(ids))
    with torch.no_grad():
        logits = m(torch.tensor([ids])).logits[0]
    store["__argmax__"] = {"out": logits.argmax(-1).float()[:, None]}
    torch.save(store, out)
    print("saved", len(store), flush=True)


def run_vllm(out, idx):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    R._shim_accelerator_empty_cache()
    R._shim_rmsnorm_gated_hidden_size()
    R._shim_gemma_rmsnorm_residual()
    R._shim_rope_contiguous()
    R._shim_platform_hybrid_kv_cache()
    R._shim_qwen_triton_warmup("skip")
    if os.environ.get("QWEN3NEXT_ZERO_PAD") == "1":
        import wy_fast_zero_pad  # noqa: F401
    ids = R._prompt_ids(MODEL)[idx]
    llm = LLM(model=MODEL, dtype="float16", max_num_seqs=1, max_model_len=512,
              tensor_parallel_size=1, enforce_eager=True, async_scheduling=False,
              enable_prefix_caching=False,
              kv_cache_memory_bytes=int(os.environ.get("QWEN3NEXT_KV_BYTES", 1 << 30)))
    store = {}
    llm.apply_model(lambda m: hook_all(m, store, len(ids)))
    o = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                     SamplingParams(temperature=0, max_tokens=1))
    print("first token", o[0].outputs[0].token_ids, flush=True)
    torch.save(store, out)
    print("saved", len(store), flush=True)


def key_order(k):
    m = re.search(r"layers\.(\d+)\.(\w+)", k)
    sub = ["input_layernorm", "linear_attn", "self_attn", "post_attention_layernorm", "mlp"]
    if k.startswith("embed"):
        return (-1, 0)
    if m:
        return (int(m.group(1)), sub.index(m.group(2)) if m.group(2) in sub else 9)
    return (999, 0)


def cmp(a, b):
    A, B = torch.load(a), torch.load(b)
    print(f"{'module':40s} {'which':4s} {'nan(v)':>7s} {'|ref|max':>9s} {'maxabs':>10s} {'rel':>9s}")
    for k in sorted(set(A) & set(B) - {"__argmax__"}, key=key_order):
        for w in ("in", "out"):
            if w not in A[k] or w not in B[k]:
                continue
            r, c = A[k][w], B[k][w]
            if r.shape != c.shape:
                print(f"{k:40s} {w:4s} shape {tuple(r.shape)} vs {tuple(c.shape)}")
                continue
            nan = int(torch.isnan(c).sum())
            d = (r - c).abs().nan_to_num(float("inf")).max().item()
            rm = r.abs().max().item()
            print(f"{k:40s} {w:4s} {nan:7d} {rm:9.3e} {d:10.3e} {d / max(rm, 1e-12):9.2e}")
    print("missing in vllm:", sorted(set(A) - set(B)))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "hf":
        run_hf(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif mode == "vllm":
        run_vllm(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        cmp(sys.argv[2], sys.argv[3])

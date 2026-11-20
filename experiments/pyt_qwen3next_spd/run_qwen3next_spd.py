# Experiment: does Qwen3-Next (GDN hybrid) run in PYT eager on QAIC, and does
# ngram/suffix SpD (upstream per-position recurrent-state slots) match the
# non-SpD greedy baseline token for token?
#
# Usage:
#   python -u run_qwen3next_spd.py child <method> <out.json> <model> [--bypass-hybrid] [--max-num-seqs N]
#   python -u run_qwen3next_spd.py driver <model> <tag> [methods...] [--bypass-hybrid] [--max-num-seqs N]
# methods: hf (CPU fp32 reference), baseline, baseline2, ngram, suffix
# "driver" runs each method in its own subprocess. The engine runs in-process
# (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the GDN probe and the bypass apply to
# the code that actually executes the model.
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TOKENS = 64
K = 4

PROMPTS = [
    # repetitive -> ngram/suffix should match and be accepted
    "Once upon a time, there was a little cat. The cat liked to play. The cat liked to play. The cat liked to",
    "Tom said, \"I am happy.\" Sue said, \"I am happy.\" Mom said, \"I am",
    "Lily had a red ball, a red hat, and a red",
    "The dog ran and ran. The dog ran and ran. Then the dog",
    # non-repetitive -> many zero-draft steps
    "Once upon a time, in a small village, there lived a girl named Mia who loved",
    "One day, a boy named Ben found a shiny key under a big tree.",
    "Sara wanted to bake a cake for her mom's birthday, but",
    "The sun was hot and the birds were singing. Max went to the park and",
]

SPEC = {
    "baseline": None,
    "baseline2": None,
    "ngram": {
        "method": "ngram",
        "num_speculative_tokens": K,
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 4,
    },
    "suffix": {"method": "suffix", "num_speculative_tokens": K},
}

_COUNTERS = {
    "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
}


def _install_gdn_probe(probe_path):
    """Log, per GDN metadata build (one per forward step), what each batch row
    receives: draft count (-1 => non-spec path), raw num_accepted_tokens from
    the previous step, query length, and the first state-slot block ids."""
    from vllm.v1.attention.backends import gdn_attn

    cls = gdn_attn.GDNAttentionMetadataBuilder
    orig = cls.build
    seen = {"last_key": None}

    def wrapped(self, common_prefix_len, common_attn_metadata,
                num_accepted_tokens=None, num_decode_draft_tokens_cpu=None,
                fast_build=False):
        md = orig(self, common_prefix_len, common_attn_metadata,
                  num_accepted_tokens=num_accepted_tokens,
                  num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                  fast_build=fast_build)
        m = common_attn_metadata
        n = int(m.num_reqs)
        qsl = m.query_start_loc_cpu[: n + 1].tolist()
        # all GDN layers share one builder call per step per kv group; dedupe
        key = (id(m), n, tuple(qsl))
        if key == seen["last_key"]:
            return md
        seen["last_key"] = key
        rec = {
            "n": n,
            "qlen": [qsl[i + 1] - qsl[i] for i in range(n)],
            "seq_lens": m.seq_lens[:n].cpu().tolist(),
            "draft": (num_decode_draft_tokens_cpu[:n].tolist()
                      if num_decode_draft_tokens_cpu is not None else None),
            "nat_raw": (num_accepted_tokens[:n].cpu().tolist()
                        if num_accepted_tokens is not None else None),
            "num_spec_decodes": int(getattr(md, "num_spec_decodes", 0)),
            "num_decodes": int(getattr(md, "num_decodes", -1)),
            "num_prefills": int(getattr(md, "num_prefills", -1)),
            "blocks": m.block_table_tensor[:n, : K + 1].cpu().tolist(),
        }
        with open(probe_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return md

    cls.build = wrapped


def _shim_pad_drafts():
    """Experiment-only causal control: pad every ngram/suffix proposal to exactly
    K drafts (repeating the last draft, or the last sampled token). Then no
    decoding request ever takes the GDN non-spec path, so neither the
    zero-draft stale-slot read nor the mixed-batch "decode as 1-token prefill"
    route can occur. The extra drafts are verified like any others, so greedy
    output must still equal the non-SpD baseline if the spec path is correct."""
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer
    from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

    def wrap(cls, sampled_pos):
        orig = cls.propose

        def propose(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            sampled = args[sampled_pos] if len(args) > sampled_pos else kwargs["sampled_token_ids"]
            padded = []
            for i, d in enumerate(out):
                d = list(d)
                last = d[-1] if d else (sampled[i][-1] if i < len(sampled) and len(sampled[i]) else None)
                if last is not None and len(d) < K:
                    d = d + [int(last)] * (K - len(d))
                padded.append(d)
            return padded
        cls.propose = propose

    # positional index of sampled_token_ids after self
    wrap(NgramProposer, 1)
    wrap(SuffixDecodingProposer, 2)


def _bypass_hybrid_raise():
    """Experiment-only: make ModelConfig.is_hybrid/is_attention_free read False
    only when evaluated from vllm_qaic/platform.py (the PYT SpD raise). All
    other readers, including the GPU runner's accepted-token logic, see the
    real value."""
    from vllm.config.model import ModelConfig

    for name in ("is_hybrid", "is_attention_free"):
        prop = ModelConfig.__dict__[name]

        def make(p):
            def getter(self):
                f = sys._getframe(1)
                if f.f_code.co_filename.endswith(os.path.join("vllm_qaic", "platform.py")):
                    return False
                return p.fget(self)
            return property(getter)

        setattr(ModelConfig, name, make(prop))


def _shim_accelerator_empty_cache():
    """Experiment-only: torch_qaic's allocator is not a c10 DeviceAllocator, so
    the generic torch.accelerator.empty_cache() (called by GDN's prefill-kernel
    warmup during profile_run) hits an INTERNAL ASSERT. Route it to the qaic
    allocator, as vllm_qaic already does for its own empty_cache call sites."""
    import torch
    import torch_qaic  # noqa: F401

    torch.accelerator.empty_cache = lambda: torch.qaic.empty_cache()


def _shim_rmsnorm_gated_hidden_size():
    """Experiment-only: vllm_qaic QAicRMSNormGated.forward_oot reads
    self.hidden_size, which upstream RMSNormGated (0.30.0) never sets.
    Equivalent source fix: use self.weight.shape[-1]."""
    from vllm_qaic.ops.layernorm import QAicRMSNormGated

    if "hidden_size" not in QAicRMSNormGated.__dict__:
        QAicRMSNormGated.hidden_size = property(lambda self: self.weight.shape[-1])


def _shim_gemma_rmsnorm_residual():
    """Experiment-only: vllm_qaic QAicGemmaRMSNorm.forward_oot's residual path
    passes the raw zero-centred self.weight to the Hexagon fused add+norm
    kernel instead of (1 + weight). Affects every GemmaRMSNorm model with a
    residual (Gemma*, Qwen3-Next). Equivalent source fix: pass 1.0 + self.weight."""
    from vllm_qaic._custom_ops import rms_norm_hexagon
    from vllm_qaic.ops.layernorm import QAicGemmaRMSNorm

    orig = QAicGemmaRMSNorm.forward_oot

    def forward_oot(self, x, residual=None):
        if residual is None:
            return orig(self, x, residual)
        return rms_norm_hexagon(residual, x, 1.0 + self.weight, self.variance_epsilon)

    QAicGemmaRMSNorm.forward_oot = forward_oot


def _shim_rope_contiguous():
    """Experiment-only: torch.ops.qaic.rotary_embedding gives wrong results for a
    non-contiguous multi-head input, which QAicRotaryEmbedding passes whenever
    rotary_dim < head_size (query[..., :rotary_dim]). Equivalent source fix:
    .contiguous() the rot slice in QAicApplyRotaryEmb.forward_static."""
    from vllm_qaic.ops.rotary_embedding import QAicApplyRotaryEmb

    orig = QAicApplyRotaryEmb.forward_static

    def forward_static(x, cos, sin, is_neox_style=True, enable_fp32_compute=False):
        return orig(x.contiguous(), cos, sin, is_neox_style, enable_fp32_compute)

    QAicApplyRotaryEmb.forward_static = staticmethod(forward_static)


def _shim_tl_load_zero():
    """Experiment-only (hypothesis test): make every Triton tl.load deterministic
    on out-of-range lanes -- padding_option="zero" for boundary-checked block-pointer
    loads, other=0 for masked loads without an explicit `other`. On QAIC those
    lanes otherwise hold uninitialized memory (see wy_fast k-load finding). The
    Triton cache key does not cover this, so run with a fresh TRITON_CACHE_DIR."""
    import triton.language as tl
    import triton.language.core as tlc

    orig = tlc.load

    @tlc.builtin
    def load(pointer, mask=None, other=None, boundary_check=(), padding_option="",
             cache_modifier="", eviction_policy="", volatile=False, _semantic=None):
        if boundary_check and not padding_option:
            padding_option = "zero"
        if mask is not None and other is None:
            other = 0
        return orig(pointer, mask, other, boundary_check, padding_option,
                    cache_modifier, eviction_policy, volatile, _semantic=_semantic)

    tlc.load = load
    tl.load = load


def _shim_platform_hybrid_kv_cache():
    """Experiment-only: QaicPlatform inherits Platform.support_hybrid_kv_cache()
    == False, which forces disable_hybrid_kv_cache_manager and makes
    unify_hybrid_kv_cache_specs fail on Mamba+attention models. CUDA/ROCm/XPU/CPU
    all return True."""
    from vllm_qaic.platform import QaicPlatform

    QaicPlatform.support_hybrid_kv_cache = classmethod(lambda cls: True)


def _shim_qwen_triton_warmup(mode):
    """Experiment-only: vLLM's qwen_triton_warmup (JIT pre-compilation on dummy
    inputs) faults the QAIC device. mode=skip -> no-op; mode=localize -> run
    each sub-warmup followed by torch.qaic.synchronize() and print which one
    faults."""
    import torch
    from vllm.model_executor.warmup import kernel_warmup as kw
    from vllm.model_executor.warmup import qwen_triton_warmup as qw

    def localized(runner, model_config):
        cfg = qw._qwen_gdn_warmup_config(runner.compilation_config.static_forward_context)
        if cfg is None:
            print("WARMUP_LOCALIZE no cfg", flush=True)
            return
        dev = runner.device
        steps = [
            ("gated_rms_norm", lambda: qw._warm_gated_rms_norm_kernel(
                dev, cfg, max(1, int(runner.max_num_tokens)), model_config.dtype)),
            ("causal_conv1d_fwd", lambda: qw._warm_causal_conv1d_fwd_kernel(dev, cfg)),
            ("fused_post_conv", lambda: qw._warm_fused_post_conv_kernel(dev, cfg)),
            ("fused_sigmoid_gating_update",
             lambda: qw._warm_fused_sigmoid_gating_delta_rule_update_kernel(dev, cfg)),
        ]
        for name, fn in steps:
            print(f"WARMUP_LOCALIZE start {name}", flush=True)
            fn()
            torch.qaic.synchronize()
            print(f"WARMUP_LOCALIZE ok {name}", flush=True)

    if mode == "skip":
        kw.qwen_triton_warmup = lambda *a, **k: print("WARMUP skipped", flush=True)
    elif mode == "localize":
        kw.qwen_triton_warmup = localized


def _install_conv1d_trace():
    """Experiment-only: wrap the GDN module's causal_conv1d_fn to print its
    inputs (read back before the call) and synchronize after it, so a device
    fault is attributed to this kernel and not the next submission."""
    import torch
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as g

    orig = g.causal_conv1d_fn
    calls = [0]

    def desc(t):
        if not isinstance(t, torch.Tensor):
            return repr(t)
        d = f"{tuple(t.shape)} st={t.stride()} {t.dtype} {t.device} contig={t.is_contiguous()}"
        if t.numel() <= 64 and t.dtype in (torch.int32, torch.int64, torch.bool):
            d += f" val={t.cpu().tolist()}"
        return d

    def traced(*args, **kwargs):
        calls[0] += 1
        n = calls[0]
        if n <= 3:
            names = ["x", "weight", "bias"]
            for nm, a in zip(names, args):
                print(f"CONV1D_TRACE#{n} {nm}: {desc(a)}", flush=True)
            for k, v in kwargs.items():
                if k == "metadata":
                    md = {a: desc(getattr(v, a)) for a in
                          ("nums_dict", "batch_ptr", "token_chunk_offset_ptr")
                          if hasattr(v, a)}
                    print(f"CONV1D_TRACE#{n} metadata: {md}", flush=True)
                else:
                    print(f"CONV1D_TRACE#{n} {k}: {desc(v)}", flush=True)
            torch.qaic.synchronize()
            print(f"CONV1D_TRACE#{n} pre-sync ok", flush=True)
        out = orig(*args, **kwargs)
        if n <= 3:
            torch.qaic.synchronize()
            print(f"CONV1D_TRACE#{n} post-sync ok", flush=True)
        return out

    g.causal_conv1d_fn = traced


def _prompt_ids(model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    return [tok(p, add_special_tokens=True)["input_ids"] for p in PROMPTS]


def child_hf(out_path, model):
    import torch
    from transformers import AutoModelForCausalLM
    ids = _prompt_ids(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
    res = []
    with torch.no_grad():
        for p in ids:
            x = torch.tensor([p])
            out = m(x, use_cache=True)
            past = out.past_key_values
            toks = []
            nxt = out.logits[0, -1].argmax().item()
            for _ in range(MAX_TOKENS):
                toks.append(nxt)
                out = m(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[0, -1].argmax().item()
            res.append(toks)
    json.dump({"method": "hf", "model": model, "token_ids": res}, open(out_path, "w"), indent=1)
    print("CHILD_RESULT hf ok")
    return 0


def child(method, out_path, model, bypass, max_num_seqs):
    if method == "hf":
        return child_hf(out_path, model)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.metrics.reader import Counter

    probe = out_path + ".probe.jsonl"
    if os.path.exists(probe):
        os.remove(probe)
    _install_gdn_probe(probe)
    if bypass:
        _bypass_hybrid_raise()
    if os.environ.get("QWEN3NEXT_PAD_DRAFTS") == "1":
        _shim_pad_drafts()
    if os.environ.get("QWEN3NEXT_NO_EMPTY_CACHE_SHIM") != "1":
        _shim_accelerator_empty_cache()
    if os.environ.get("QWEN3NEXT_NO_NORM_SHIM") != "1":
        _shim_rmsnorm_gated_hidden_size()
    if os.environ.get("QWEN3NEXT_NO_GEMMA_SHIM") != "1":
        _shim_gemma_rmsnorm_residual()
    if os.environ.get("QWEN3NEXT_NO_ROPE_SHIM") != "1":
        _shim_rope_contiguous()
    if os.environ.get("QWEN3NEXT_TL_LOAD_ZERO") == "1":
        _shim_tl_load_zero()
    if os.environ.get("QWEN3NEXT_NO_HMA_SHIM") != "1":
        _shim_platform_hybrid_kv_cache()
    _shim_qwen_triton_warmup(os.environ.get("QWEN3NEXT_WARMUP", "skip"))
    if os.environ.get("QWEN3NEXT_CONV_TRACE") == "1":
        _install_conv1d_trace()
    if os.environ.get("QWEN3NEXT_ZERO_PAD") == "1":
        # Experiment-only: FLA recompute_w_u_fwd_kernel with padding_option="zero"
        # on its boundary-checked loads (see kernel probe chunk NaN finding).
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import wy_fast_zero_pad  # noqa: F401

    kwargs = dict(
        model=model,
        dtype="float16",
        max_num_seqs=max_num_seqs,
        max_model_len=512,
        tensor_parallel_size=1,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        **({"kv_cache_memory_bytes": int(os.environ["QWEN3NEXT_KV_BYTES"])}
           if os.environ.get("QWEN3NEXT_KV_BYTES") else {}),
        gpu_memory_utilization=0.9,
        disable_log_stats=False,
    )
    if SPEC[method] is not None:
        kwargs["speculative_config"] = dict(SPEC[method])
    llm = LLM(**kwargs)
    if os.environ.get("QWEN3NEXT_NO_TIE_SHIM") != "1":
        # Experiment-only: upstream Qwen3NextForCausalLM ignores
        # tie_word_embeddings, and this 70M checkpoint ships no lm_head.weight,
        # so lm_head would stay uninitialized. Tie it to embed_tokens.
        def _tie(m):
            m.lm_head.weight.data.copy_(m.model.embed_tokens.weight.data)
        llm.apply_model(_tie)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)

    def counters():
        r = dict.fromkeys(_COUNTERS.values(), 0)
        for m in llm.get_metrics():
            if isinstance(m, Counter) and m.name in _COUNTERS:
                r[_COUNTERS[m.name]] += m.value
        return r

    ids = _prompt_ids(model)
    before = counters()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sp)
    after = counters()
    res = {
        "method": method,
        "model": model,
        "max_num_seqs": max_num_seqs,
        "prompt_lens": [len(p) for p in ids],
        "token_ids": [list(o.outputs[0].token_ids) for o in outs],
        "texts": [o.outputs[0].text for o in outs],
        **{k: after[k] - before[k] for k in after},
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("CHILD_RESULT", json.dumps({k: v for k, v in res.items() if k not in ("texts", "token_ids")}))
    del llm
    return 0


def summarize_probe(path):
    """Hazard = a row that takes the non-spec path (draft == -1) as a decode
    (qlen == 1) while the previous step accepted > 1 token for it (nat_raw > 1).
    Rows are identified by their seq_len trajectory is not needed: we report
    per-step counts and the first hazard step index."""
    if not os.path.exists(path):
        return {"probe_file": "missing"}
    recs = [json.loads(line) for line in open(path)]
    steps_spec = sum(1 for r in recs if r["num_spec_decodes"] > 0)
    hazards = []
    zero_draft_decodes = 0
    for s, r in enumerate(recs):
        if r["draft"] is None or r["nat_raw"] is None:
            continue
        for i in range(r["n"]):
            if r["draft"][i] == -1 and r["qlen"][i] == 1:
                zero_draft_decodes += 1
                if r["nat_raw"][i] > 1:
                    hazards.append({"step": s, "row": i, "nat": r["nat_raw"][i],
                                    "seq_len": r["seq_lens"][i], "blocks": r["blocks"][i]})
    return {
        "steps": len(recs),
        "steps_with_spec_decodes": steps_spec,
        "zero_draft_decode_rows": zero_draft_decodes,
        "hazard_rows": len(hazards),
        "hazard_samples": hazards[:8],
        "block_shape_sample": [len(recs[-1]["blocks"]), len(recs[-1]["blocks"][0])] if recs and recs[-1]["blocks"] else None,
    }


def first_div(a, b):
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if d is None and len(a) != len(b):
        d = min(len(a), len(b))
    return d


def driver(model, tag, methods, bypass, max_num_seqs):
    results = {}
    for m in methods:
        out = os.path.join(HERE, f"{tag}_{m}.json")
        log = os.path.join(HERE, f"{tag}_{m}.log")
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "child", m, out, model,
               "--max-num-seqs", str(max_num_seqs)]
        if bypass:
            cmd.append("--bypass-hybrid")
        env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0")
        if m == "hf":
            env["CUDA_VISIBLE_DEVICES"] = ""
        print("RUN", " ".join(cmd), ">", log, flush=True)
        with open(log, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        print(f"  rc={rc}", flush=True)
        if rc != 0 or not os.path.exists(out):
            results[m] = {"error": f"rc={rc}, see {log}"}
            continue
        results[m] = json.load(open(out))
        results[m]["probe"] = summarize_probe(out + ".probe.jsonl")
    summary = {}
    base = results.get("baseline")
    for m, r in results.items():
        s = {k: r.get(k) for k in ("error", "num_draft_tokens", "num_accepted_tokens", "probe")}
        if base and "token_ids" in base and "token_ids" in r and m != "baseline":
            s["vs_baseline_equal"] = [a == b for a, b in zip(base["token_ids"], r["token_ids"])]
            s["vs_baseline_first_div"] = [first_div(a, b) for a, b in zip(base["token_ids"], r["token_ids"])]
        summary[m] = s
    print("SUMMARY", json.dumps(summary, indent=1))
    with open(os.path.join(HERE, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    bypass = "--bypass-hybrid" in a
    a = [x for x in a if x != "--bypass-hybrid"]
    mns = 8
    if "--max-num-seqs" in a:
        i = a.index("--max-num-seqs")
        mns = int(a[i + 1])
        del a[i : i + 2]
    if a[0] == "child":
        sys.exit(child(a[1], a[2], a[3], bypass, mns))
    driver(a[1], a[2], a[3:] or ["hf", "baseline", "baseline2", "ngram", "suffix"], bypass, mns)

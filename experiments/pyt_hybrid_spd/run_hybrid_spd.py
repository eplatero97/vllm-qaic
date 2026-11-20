# Experiment: does hybrid (mamba2 + attention) SpD in PYT eager mode on QAIC
# match the non-SpD greedy baseline?
#
# Usage:
#   python -u run_hybrid_spd.py child <method> <out.json> [model] [--bypass-attn-free]
#   python -u run_hybrid_spd.py driver [model] [methods...]
# "driver" runs each method in its own subprocess and compares token ids.
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "ibm-granite/granite-4.0-h-350m"

PROMPTS = [
    # repetitive -> ngram/suffix should match and be accepted
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. The cat sat on the",
    # repetitive list whose natural continuation diverges from the pattern
    "Count: 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5. Now count to twenty: 1 2 3 4 5 6",
    # repetitive code, continuation likely differs from prior lines
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n",
    # short, non-repetitive
    "My name is",
    # medium, non-repetitive
    "Explain in simple terms why the sky appears blue during the day and red at sunset.",
    # repeated phrase followed by a divergent twist
    "Roses are red, violets are blue. Roses are red, violets are blue. Roses are red, but grass is",
]

SPEC = {
    "baseline": None,
    "ngram": {
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 4,
    },
    "suffix": {"method": "suffix", "num_speculative_tokens": 5},
}

_COUNTERS = {
    "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
}


def _install_mamba_probe(probe_path):
    """Log what the mamba metadata builder receives. Patched in the parent
    before LLM() so a forked EngineCore inherits it; each process appends
    JSON lines tagged with its pid."""
    import torch
    from vllm.v1.attention.backends import mamba_attn

    cls = mamba_attn.BaseMambaAttentionMetadataBuilder
    orig = cls._compute_common_metadata

    def wrapped(self, common_attn_metadata, *, num_accepted_tokens=None, **kw):
        md = orig(
            self, common_attn_metadata, num_accepted_tokens=num_accepted_tokens, **kw
        )
        rec = {
            "pid": os.getpid(),
            "builder": type(self).__name__,
            "use_spec_decode": bool(getattr(self, "use_spec_decode", False)),
            "nat_is_none": num_accepted_tokens is None,
            "num_decodes": int(getattr(md, "num_decodes", -1)),
            "num_prefills": int(getattr(md, "num_prefills", -1)),
            "md_nat_is_none": getattr(md, "num_accepted_tokens", None) is None,
        }
        if num_accepted_tokens is not None:
            try:
                rec["nat"] = num_accepted_tokens.detach().to("cpu").tolist()
            except Exception as e:  # noqa: BLE001
                rec["nat_err"] = repr(e)
        sidx = getattr(md, "state_indices_tensor_d", None)
        if isinstance(sidx, torch.Tensor):
            rec["state_indices_d_shape"] = list(sidx.shape)
        with open(probe_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return md

    cls._compute_common_metadata = wrapped


def _bypass_attention_free_raise():
    """Experiment-only: let attention-free models through the PYT SpD check by
    hiding is_attention_free during QaicPlatform.check_and_update_config."""
    from vllm_qaic.platform import QaicPlatform

    orig = QaicPlatform.check_and_update_config.__func__

    @classmethod
    def patched(cls, vllm_config):
        mc = vllm_config.model_config
        mcls = type(mc)
        prop = mcls.__dict__.get("is_attention_free")
        saved = None
        if isinstance(prop, property):
            mcls.is_attention_free = property(lambda self: False)
        else:
            saved = mc.__dict__.get("is_attention_free")
            object.__setattr__(mc, "is_attention_free", False)
        try:
            return orig(cls, vllm_config)
        finally:
            if isinstance(prop, property):
                mcls.is_attention_free = prop
            elif saved is not None:
                object.__setattr__(mc, "is_attention_free", saved)

    QaicPlatform.check_and_update_config = patched


def child(method, out_path, model, bypass):
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter

    probe = out_path + ".probe.jsonl"
    if os.path.exists(probe):
        os.remove(probe)
    _install_mamba_probe(probe)
    if bypass:
        _bypass_attention_free_raise()

    kwargs = dict(
        model=model,
        dtype="float16",
        max_num_seqs=8,
        max_model_len=512,
        tensor_parallel_size=1,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.9,
        disable_log_stats=False,
    )
    if SPEC[method] is not None:
        kwargs["speculative_config"] = dict(SPEC[method])
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=48)

    def counters():
        r = dict.fromkeys(_COUNTERS.values(), 0)
        for m in llm.get_metrics():
            if isinstance(m, Counter) and m.name in _COUNTERS:
                r[_COUNTERS[m.name]] += m.value
        return r

    before = counters()
    outs = llm.generate(PROMPTS, sp)
    after = counters()
    res = {
        "method": method,
        "model": model,
        "token_ids": [list(o.outputs[0].token_ids) for o in outs],
        "texts": [o.outputs[0].text for o in outs],
        **{k: after[k] - before[k] for k in after},
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("CHILD_RESULT", json.dumps({k: v for k, v in res.items() if k != "texts"}))
    del llm
    return 0


def summarize_probe(path):
    if not os.path.exists(path):
        return {"probe_file": "missing"}
    recs = [json.loads(line) for line in open(path)]
    nn = [r for r in recs if not r["nat_is_none"]]
    multi = [r for r in nn if any(x > 1 for x in r.get("nat", []))]
    shapes = sorted({tuple(r["state_indices_d_shape"]) for r in recs if "state_indices_d_shape" in r})
    return {
        "calls": len(recs),
        "calls_nat_non_none": len(nn),
        "calls_with_nat_gt1": len(multi),
        "use_spec_decode_any": any(r["use_spec_decode"] for r in recs),
        "pids": sorted({r["pid"] for r in recs}),
        "state_indices_d_shapes": [list(s) for s in shapes][:6],
        "nat_samples": [r["nat"] for r in multi[:5]],
    }


def driver(model, methods, bypass):
    tag = model.split("/")[-1]
    results = {}
    for m in methods:
        out = os.path.join(HERE, f"{tag}_{m}.json")
        log = os.path.join(HERE, f"{tag}_{m}.log")
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "child", m, out, model]
        if bypass:
            cmd.append("--bypass-attn-free")
        print("RUN", " ".join(cmd), ">", log, flush=True)
        with open(log, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        print(f"  rc={rc}", flush=True)
        if rc != 0 or not os.path.exists(out):
            results[m] = {"error": f"rc={rc}, see {log}"}
            if m == "baseline":
                break
            continue
        results[m] = json.load(open(out))
        results[m]["probe"] = summarize_probe(out + ".probe.jsonl")
    summary = {}
    base = results.get("baseline")
    for m, r in results.items():
        s = {k: r.get(k) for k in ("error", "num_draft_tokens", "num_accepted_tokens", "probe")}
        if base and "token_ids" in base and "token_ids" in r and m != "baseline":
            eq = [a == b for a, b in zip(base["token_ids"], r["token_ids"])]
            first_div = []
            for a, b in zip(base["token_ids"], r["token_ids"]):
                d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                if d is None and len(a) != len(b):
                    d = min(len(a), len(b))
                first_div.append(d)
            s["per_request_equal"] = eq
            s["first_divergence_idx"] = first_div
            dr, ac = r.get("num_draft_tokens", 0), r.get("num_accepted_tokens", 0)
            s["0<accepted<drafted"] = 0 < ac < dr
        summary[m] = s
    print("SUMMARY", json.dumps(summary, indent=1))
    with open(os.path.join(HERE, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    bypass = "--bypass-attn-free" in a
    a = [x for x in a if x != "--bypass-attn-free"]
    if a[0] == "child":
        sys.exit(child(a[1], a[2], a[3] if len(a) > 3 else DEFAULT_MODEL, bypass))
    model = a[1] if len(a) > 1 else DEFAULT_MODEL
    methods = a[2:] or ["baseline", "ngram", "suffix"]
    driver(model, methods, bypass)

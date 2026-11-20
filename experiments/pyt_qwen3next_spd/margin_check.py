"""At each vLLM-vs-HF first divergence, report HF fp32's top-2 logit margin, and the
rank of vLLM's chosen token under HF (near-tie => fp16 drift plausible)."""
import json
import sys

import torch
from transformers import AutoModelForCausalLM

import run_qwen3next_spd as R

MODEL = "theo77186/Qwen3-Next-70M-TinyStories"
hf = json.load(open(sys.argv[1]))
vl = json.load(open(sys.argv[2]))
prompts = R._prompt_ids(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
for i, (a, b) in enumerate(zip(hf["token_ids"], vl["token_ids"])):
    d = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if d is None:
        continue
    with torch.no_grad():
        lg = m(torch.tensor([prompts[i] + a[:d]])).logits[0, -1]
    top = lg.topk(5)
    rank = int((lg > lg[b[d]]).sum())
    print(f"prompt {i} div@{d}: hf top2 logits {top.values[0]:.4f} {top.values[1]:.4f} "
          f"margin={top.values[0] - top.values[1]:.4f}  vllm tok rank under HF={rank} "
          f"(gap {lg[a[d]] - lg[b[d]]:.4f})", flush=True)

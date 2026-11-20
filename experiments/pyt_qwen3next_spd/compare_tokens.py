"""Compare per-prompt greedy token ids: python compare_tokens.py ref.json cand.json"""
import json
import sys

ref = json.load(open(sys.argv[1]))["token_ids"]
cand = json.load(open(sys.argv[2]))["token_ids"]
print(f"ref={sys.argv[1]} cand={sys.argv[2]}")
for i, (r, c) in enumerate(zip(ref, cand)):
    n = min(len(r), len(c))
    div = next((j for j in range(n) if r[j] != c[j]), None)
    match = sum(a == b for a, b in zip(r, c))
    print(f"prompt {i}: len ref/cand={len(r)}/{len(c)} first_div={div if div is not None else 'none'} "
          f"positional_match={match}/{n}")

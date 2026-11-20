"""Offline hazard analysis of a GDN probe log (run_qwen3next_spd.py writes it).

The probe logs one record per GDN metadata build. A step has one build per
KV-cache group, so the records for one step are consecutive. When no request in
the batch has drafts, draft/nat_raw are None.

Per request we track the computed-token count C = seq_len - qlen. A request
gets a step's tokens and accepts a = C_next - C_prev of them.
Hazard: a step that is a plain decode on the non-spec path (draft None or -1,
qlen 1) right after a spec step for the same request that accepted a > 1. The
spec step left the correct recurrent state in slot a-1, but this decode reads
block_table[:, 0].
The decode at computed C produces output index C - prompt_len + 1. That is the
first output index that may be wrong.

Usage: hazard_analyze.py probe.jsonl run.json baseline.json
"""
import json, sys

probe, run, base = sys.argv[1:4]
recs = [json.loads(l) for l in open(probe)]
R = json.load(open(run)); B = json.load(open(base))
P = R["prompt_lens"]

# Every KV-cache group's build sees each request once per step, under a
# group-specific block id. Track each block id as its own chain. A prefill
# (C == 0) on a chain starts a new request. The per-group copies of one request
# have identical histories; dedupe them, then map requests to prompts in order
# of first appearance (prefill order == prompt order).
chains = {}        # bid -> current request key
reqs = {}          # (bid, first_rec_idx) -> list of (C, qlen, spec_flag)
for ri, r in enumerate(recs):
    for i in range(r["n"]):
        bid = r["blocks"][i][0]
        q = r["qlen"][i]; C = r["seq_lens"][i] - q
        if C == 0:
            chains[bid] = (bid, ri); reqs[chains[bid]] = []
        d = r["draft"][i] if r["draft"] is not None else -1
        reqs[chains[bid]].append((C, q, d >= 0))
# Cluster the per-group copies of one request: their prefills are logged within
# a few records of each other. Keep the longest copy; a copy can lose a record
# when a group's block id is reused across requests.
G = sum(1 for r in recs[:16] if r["num_prefills"] and r["seq_lens"][0] == r["qlen"][0]
        and r["seq_lens"] == recs[0]["seq_lens"])
G = max(G, 1)
keys = sorted(reqs, key=lambda k: k[1])
uniq = []
while keys:
    # One request per prompt per group, so a cluster is the next G keys whose
    # first record has the same (seq_len, qlen) set; group them by that.
    k0 = keys[0]
    sig = (tuple(recs[k0[1]]["seq_lens"]), tuple(recs[k0[1]]["qlen"]))
    row = next(i for i in range(recs[k0[1]]["n"]) if recs[k0[1]]["blocks"][i][0] == k0[0])
    clus = [k for k in keys if abs(k[1] - k0[1]) < 4 * G
            and recs[k[1]]["seq_lens"][[i for i in range(recs[k[1]]["n"]) if recs[k[1]]["blocks"][i][0] == k[0]][0]]
                == recs[k0[1]]["seq_lens"][row]
            and reqs[k][:3] == reqs[k0][:3]][:G]
    for k in clus:
        keys.remove(k)
    uniq.append(max((reqs[k] for k in clus), key=len))
assert len(uniq) == len(P), (len(uniq), len(P), G)
hist = dict(enumerate(uniq))

def first_div(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)

out = []
for p in sorted(hist):
    h = hist[p]
    assert h[0][0] == 0 and h[0][1] == P[p], (p, h[0], P[p])
    hazards = []
    for j in range(1, len(h) - 1):
        C, q, spec = h[j]; Cn = h[j + 1][0]
        a = Cn - C
        if spec and a > 1 and not h[j + 1][2] and h[j + 1][1] == 1:
            hazards.append({"after_spec_C": C, "accepted": a, "out_idx": Cn - P[p] + 1})
    fd = first_div(B["token_ids"][p], R["token_ids"][p])
    nspec = sum(1 for x in h if x[2])
    fh = hazards[0]["out_idx"] if hazards else None
    out.append(dict(prompt=p, first_div=fd, first_hazard_out_idx=fh,
                    n_hazards=len(hazards), spec_steps=nspec))
    print(f"prompt {p}: first_div={fd} first_hazard_out_idx={fh} "
          f"n_hazards={len(hazards)} spec_steps={nspec} hazards={hazards[:4]}")

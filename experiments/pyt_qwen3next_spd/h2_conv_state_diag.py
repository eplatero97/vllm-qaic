# Diagnose (a)'s conv-state writeback: per-slot / per-position comparison of
# what causal_conv1d_fn wrote vs expected [old[1], old[2], x0] for 1-token seqs.
# Usage: python -u h2_conv_state_diag.py   (H2_NO_TL_ZERO=1 to disable tl.load shim)
import torch
import h2_prefill_as_decode_probe as P

for lens, meta in (([1, 1], True), ([1, 1], False), ([1, 1, 3, 1], True), ([2, 5], True)):
    B = len(lens); idx = P.SLOTS[:B]
    conv_sd, ssm, mixed, a, b = P.make_inputs(sum(lens), 300)
    cd, sd = P.to_dev(conv_sd, ssm)
    qsl = torch.zeros(B + 1, dtype=torch.int32); qsl[1:] = torch.tensor(lens).cumsum(0)
    m = None
    if meta:
        from types import SimpleNamespace
        nd, bp, tp = P.compute_causal_conv1d_metadata(qsl, device=P.dev)
        m = SimpleNamespace(nums_dict=nd, batch_ptr=bp, token_chunk_offset_ptr=tp)
    P.causal_conv1d_fn(mixed.to(P.dev).transpose(0, 1), P.conv_w_d, None, activation="silu",
                       conv_states=cd.transpose(-1, -2),
                       has_initial_state=torch.ones(B, dtype=torch.bool).to(P.dev),
                       cache_indices=torch.tensor(idx, dtype=torch.int32).to(P.dev),
                       query_start_loc=qsl.to(P.dev), metadata=m)
    P.sync(); new = cd.float().cpu(); old = conv_sd.float()
    print(f"== lens={lens} metadata={'builder' if meta else 'None'}")
    s = 0
    for r, L in enumerate(lens):
        sl = idx[r]; seq = torch.cat([old[sl, :3], mixed[s:s+L].float()]); exp = seq[-3:]
        for pos in range(3):
            got = new[sl, pos]
            cands = {"exp": exp[pos], "old[pos]": old[sl, pos], "zero": torch.zeros_like(got)}
            for j in range(L): cands[f"x{j}"] = mixed[s+j].float()
            for j in range(P.S): cands[f"old[{j}]"] = old[sl, j]
            best = min(cands, key=lambda k: (got - cands[k]).abs().max().item())
            print(f"  req{r} L={L} slot{sl} pos{pos}: |got-exp|={(got-exp[pos]).abs().max():.3e} "
                  f"closest={best} ({(got-cands[best]).abs().max():.3e}) nan={torch.isnan(got).sum().item()}")
        s += L

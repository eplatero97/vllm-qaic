# Isolate the Mamba2 SSD Triton kernel that faults on QAIC.
# Usage: python -u kernel_probe.py <cfg_idx|all> [stage]
#   Forces _chunk_state_fwd_kernel to a single autotune config (or all), then
#   runs the stages of the SSD prefill path at granite-4.0-h-350m warmup shapes
#   (seqlen=chunk=256, nheads=48, hdim=32, ngroups=1, dstate=128, fp16).
#   Results are compared against a float32 torch reference on CPU.
import sys
import time

import torch
import torch_qaic  # noqa: F401

from vllm.model_executor.layers.mamba.ops import ssd_chunk_state as cs

dev = torch.device("qaic:0")
cfg_arg = sys.argv[1] if len(sys.argv) > 1 else "0"
if cfg_arg != "all":
    cfg = cs._chunk_state_fwd_kernel.configs[int(cfg_arg)]
    cs._chunk_state_fwd_kernel.configs = [cfg]
    print("forced chunk_state config:", cfg, flush=True)

torch.manual_seed(0)
L, H, P, G, N, CH = 256, 48, 32, 1, 128, 256
if len(sys.argv) > 2 and sys.argv[2] == "tiny":
    L, H, P, G, N, CH = 32, 1, 32, 1, 32, 32
print("shapes L,H,P,G,N,CH =", (L, H, P, G, N, CH), flush=True)
x = torch.randn(L, H, P, dtype=torch.float16)
dt = torch.randn(L, H, dtype=torch.float16)
A = -torch.rand(H, dtype=torch.float32)
B = torch.randn(L, G, N, dtype=torch.float16)
dt_bias = torch.randn(H, dtype=torch.float32)
cu = torch.tensor([0, L], dtype=torch.int32)

t = time.time()
dA_cs, dt_out = cs._chunk_cumsum_fwd(
    dt.to(dev), A.to(dev), CH, cu.to(dev), dt_bias=dt_bias.to(dev),
    dt_softplus=True, dt_limit=(0.0, float("inf")),
)
torch.qaic.synchronize()
print(f"chunk_cumsum OK ({time.time()-t:.1f}s)", flush=True)

# reference cumsum
dtr = torch.nn.functional.softplus(dt.float() + dt_bias).T.reshape(H, 1, L)
dAr = torch.cumsum(dtr * A.view(H, 1, 1), dim=-1)
print("cumsum max abs err:", (dA_cs.cpu() - dAr).abs().max().item(),
      "dt err:", (dt_out.cpu() - dtr).abs().max().item(), flush=True)

t = time.time()
states = cs._chunk_state_fwd(B.to(dev), x.to(dev), dt_out, dA_cs, cu.to(dev),
                             states_in_fp32=True)
torch.qaic.synchronize()
print(f"chunk_state OK ({time.time()-t:.1f}s)", flush=True)
# ref: states[c,h,p,n] = sum_l x[l,h,p] * B[l,g,n] * exp(dA[-1]-dA[l]) * dt[l]
decay = torch.exp(dAr[:, 0, -1:] - dAr[:, 0, :]) * dtr[:, 0, :]  # (H, L)
ref = torch.einsum("lhp,ln,hl->hpn", x.float(), B[:, 0].float(), decay)
print("chunk_state max abs err:", (states.cpu()[0] - ref).abs().max().item(),
      "ref max:", ref.abs().max().item(), flush=True)

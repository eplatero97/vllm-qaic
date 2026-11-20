"""torch.ops.qaic.rotary_embedding on contiguous vs strided (partial-rotary slice) input."""
import torch
import torch_qaic  # noqa: F401
from vllm.platforms import current_platform

current_platform.device_type  # noqa: B018  (initialize platform before vllm_qaic imports)
import vllm_qaic._custom_ops  # noqa: F401  (registers qaic ops)

torch.manual_seed(0)
T, D, R = 27, 256, 64


def ref(x, c, s):
    x1, x2 = x.float().chunk(2, -1)
    c, s = c.float()[:, None], s.float()[:, None]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1)


c = torch.randn(T, R // 2).half()
s = torch.randn(T, R // 2).half()
for H in (1, 2, 4):
    full = torch.randn(T, H, D).half()
    for name, x in (("strided", full[..., :R]), ("contig", full[..., :R].contiguous())):
        xq = full.to("qaic")[..., :R] if name == "strided" else x.to("qaic")
        out = torch.ops.qaic.rotary_embedding(xq, c.to("qaic"), s.to("qaic"), True).cpu().float()
        r = ref(x, c, s)
        print(f"H={H} {name:8s} is_contig={xq.is_contiguous()} stride={xq.stride()} "
              f"rel={(out - r).abs().max() / r.abs().max():.2e}", flush=True)

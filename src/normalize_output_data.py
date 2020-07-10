import torch

#torch.jit.optimized_execution(True)


def wrap(data: torch.Tensor):
    start = data[:, :, :6]
    mid = data[:, :, 6]

    max0 = torch.where(start < 1000, start, torch.zeros_like(start))
    max0 = max0.max(dim=1, keepdim=True)[0]
    max0.clamp_(min=1)

    max1 = torch.where(mid < 1000, mid, torch.zeros_like(mid))
    max1 = max1.max(dim=1, keepdim=True)[0]
    max1.clamp_(min=1)

    min_mid = torch.where(mid >= 0, mid, torch.zeros_like(mid))
    min_obs = min_mid.min(dim=1, keepdim=True)[0]

    mid.sub_(min_obs)
    max1.sub_(min_obs)
    mid.div_(max1)

    start.div_(max0)

    data.clamp_(-1, 1)

    data[:, :, :6].sub_(data[:, :, :6].mean())
    data[:, :, 7:].sub_(data[:, :, 7:].mean())
    data.detach_()

wrap = torch.jit.script(wrap)

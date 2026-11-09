import torch


@torch.no_grad()
def ema_update(online: torch.nn.Module, target: torch.nn.Module, m: float):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)
    for b_o, b_t in zip(online.buffers(), target.buffers()):
        b_t.data.copy_(b_o.data)

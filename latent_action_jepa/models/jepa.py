import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ViTEncoder
from .token_reducer import TokenReducer
from .inverse_encoder import InverseActionEncoder
from .transition import TransitionModel
from .flow_prior import FlowPrior
from .ema import ema_update


class LatentActionJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.online_encoder = ViTEncoder(cfg["encoder_name"], pretrained=cfg["pretrained"])
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        d = self.online_encoder.embed_dim
        self.state_reducer = TokenReducer(d, cfg["state_dim"], cfg["state_tokens"], depth=2)
        self.target_future_reducer = TokenReducer(d, cfg["state_dim"], cfg["future_tokens"], depth=2)
        # Also keep an online "future_reducer" we can EMA from -> target_future_reducer
        self.online_future_reducer = copy.deepcopy(self.target_future_reducer)
        for p in self.target_future_reducer.parameters():
            p.requires_grad_(False)

        self.inverse_encoder = InverseActionEncoder(
            cfg["state_dim"], cfg["action_dim"], cfg["action_tokens"], depth=cfg["inverse_depth"],
        )
        self.transition_model = TransitionModel(
            cfg["state_dim"], cfg["action_dim"], cfg["future_tokens"], depth=cfg["transition_depth"],
        )
        self.flow_model = FlowPrior(cfg["state_dim"], cfg["action_dim"], depth=cfg["flow_depth"])

        self.z_dropout = cfg.get("z_dropout", 0.1)
        self.z_noise_std = cfg.get("z_noise_std", 0.0)

    def encode_current(self, x):
        toks = self.online_encoder(x)
        return self.state_reducer(toks)

    @torch.no_grad()
    def encode_target_future(self, x):
        toks = self.target_encoder(x)
        return self.target_future_reducer(toks)

    def perturb_z(self, z, train=True):
        if train and self.z_noise_std > 0:
            z = z + self.z_noise_std * torch.randn_like(z)
        if train and self.z_dropout > 0:
            mask = (torch.rand_like(z[..., :1]) > self.z_dropout).float()
            z = z * mask
        return z

    def forward_phase1(self, x_t, x_tp, train=True):
        h_t = self.encode_current(x_t)
        h_tp = self.encode_target_future(x_tp)
        z_star = self.inverse_encoder(h_t, h_tp)
        z_in = self.perturb_z(z_star, train=train)
        h_pred = self.transition_model(h_t, z_in)
        return {"H_t": h_t, "H_tp": h_tp, "Z_star": z_star, "H_pred": h_pred}

    def flow_step(self, h_t, z_star):
        eps = torch.randn_like(z_star)
        tau = torch.rand(z_star.shape[0], 1, 1, device=z_star.device)
        z_tau = (1 - tau) * eps + tau * z_star.detach()
        v_target = z_star.detach() - eps
        v_pred = self.flow_model(z_tau, tau, h_t.detach())
        return v_pred, v_target

    @torch.no_grad()
    def sample_flow(self, h_t, action_tokens, action_dim, steps=20):
        b = h_t.shape[0]
        z = torch.randn(b, action_tokens, action_dim, device=h_t.device)
        for i in range(steps):
            tau = torch.full((b, 1, 1), i / steps, device=h_t.device)
            v = self.flow_model(z, tau, h_t)
            z = z + v / steps
        return z

    def update_ema(self, m):
        ema_update(self.online_encoder, self.target_encoder, m)
        ema_update(self.online_future_reducer, self.target_future_reducer, m)


def cosine_distance(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(-1)).mean()

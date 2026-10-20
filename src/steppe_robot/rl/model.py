from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from ncps.torch import CfC


def log_tanh_jacobian(z: torch.Tensor) -> torch.Tensor:
    return 2 * (math.log(2) - z - F.softplus(-2 * z))


def action_log_prob(z: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return (Normal(mean, log_std.exp()).log_prob(z) - log_tanh_jacobian(z)).sum(-1)


def action_entropy(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    # Differentiable Monte Carlo estimate of entropy AFTER tanh, including its Jacobian.
    distribution = Normal(mean, log_std.exp())
    return distribution.entropy().sum(-1) + log_tanh_jacobian(distribution.rsample()).sum(-1)


class CfCActorCritic(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.camera = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=3),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(384, 128),
            nn.ELU(),
        )
        self.lidar = nn.Sequential(
            nn.Conv1d(3, 16, 7, stride=3, padding=3, padding_mode="circular"),
            nn.ELU(),
            nn.Conv1d(16, 32, 5, stride=3, padding=2, padding_mode="circular"),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ELU(),
        )
        self.proprio = nn.Sequential(nn.Linear(30, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
        self.fusion = nn.Sequential(nn.Linear(256, 192), nn.ELU())
        self.memory = CfC(
            192,
            hidden_size,
            batch_first=True,
            return_sequences=True,
            mixed_memory=False,
            backbone_units=128,
            backbone_layers=1,
            backbone_dropout=0,
        )
        self.actor = nn.Linear(hidden_size, 2)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1)
        nn.init.zeros_(self.critic.bias)

    def initial_state(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device)

    def encode(self, images: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        b, t = vectors.shape[:2]
        pixels = images.reshape(b * t, 3, 120, 160).float() / 255
        v = vectors.reshape(b * t, 1118)
        scan = torch.stack((v[:, :360] / 12, v[:, 360:720], v[:, 720:1080]), dim=1)
        # Absolute GPS position, course, simulation clock and episode clock are excluded.
        # They can expose world identity and are unnecessary for local goal following.
        p = torch.cat(
            (
                (v[:, 1080:1083] / 9.81).clamp(-5, 5),
                (v[:, 1083:1086] / 5).clamp(-5, 5),
                (v[:, 1086:1090] / 120).clamp(-5, 5),
                (v[:, 1090:1091] / 30).clamp(0, 4),
                v[:, 1091:1093],
                (v[:, 1095:1101] / 0.1).clamp(0, 10),
                v[:, 1101:1105],
                (v[:, 1105:1106] / 0.1).clamp(0, 10),
                (v[:, 1109:1112].norm(dim=-1, keepdim=True) / 3).clamp(0, 5),
                v[:, 1113:1116],
                v[:, 1116:1117] / 0.02,
                v[:, 1117:1118] / 0.1,
            ),
            dim=-1,
        )
        return self.fusion(torch.cat((self.camera(pixels), self.lidar(scan), self.proprio(p)), -1)).reshape(b, t, 192)

    def forward_sequence(self, images, vectors, state, episode_starts, valid=None, burn_in=0):
        b, t = vectors.shape[:2]
        if valid is None:
            valid = torch.ones((b, t), dtype=torch.bool, device=vectors.device)
        if burn_in:
            with torch.no_grad():
                prefix = self.encode(images[:, :burn_in], vectors[:, :burn_in])
            encoded = torch.cat((prefix, self.encode(images[:, burn_in:], vectors[:, burn_in:])), 1)
        else:
            encoded = self.encode(images, vectors)
        outputs = []
        for i in range(t):
            reset = (episode_starts[:, i] & valid[:, i]).unsqueeze(-1)
            state = torch.where(reset, torch.zeros_like(state), state)
            # First observation has dt=0 in the contract; initialize its cell using one nominal interval.
            dt = vectors[:, i, 1095]
            dt = torch.where(dt > 0, dt, torch.full_like(dt, 0.1)).clamp(0.001, 1)
            # ncps 1.0.1's sequence wrapper squeezes [B,1] timespans to [B],
            # which cannot broadcast over hidden features when B>1. Use its official
            # CfCCell directly and preserve [B,1] for per-sample physical time.
            if i < burn_in:
                with torch.no_grad():
                    _, candidate = self.memory.rnn_cell(encoded[:, i], state, dt[:, None])
            else:
                _, candidate = self.memory.rnn_cell(encoded[:, i], state, dt[:, None])
            state = torch.where(valid[:, i : i + 1], candidate, state)
            outputs.append(state)
        hidden = torch.stack(outputs, dim=1)
        return self.actor(hidden), self.log_std.clamp(-3, 0.5).expand(b, t, 2), self.critic(hidden).squeeze(-1), state

    @torch.no_grad()
    def act(
        self,
        image: np.ndarray,
        vector: np.ndarray,
        state: torch.Tensor,
        episode_start: bool,
        deterministic: bool = False,
    ):
        device = next(self.parameters()).device
        pixels = torch.as_tensor(image, device=device)[None, None]
        values = torch.as_tensor(vector, device=device)[None, None]
        starts = torch.tensor([[episode_start]], dtype=torch.bool, device=device)
        mean, log_std, value, next_state = self.forward_sequence(pixels, values, state, starts)
        z = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        log_prob = action_log_prob(z, mean, log_std)
        return (
            z.tanh()[0, 0].cpu().numpy(),
            z[0, 0].cpu().numpy(),
            float(log_prob[0, 0]),
            float(value[0, 0]),
            next_state,
        )

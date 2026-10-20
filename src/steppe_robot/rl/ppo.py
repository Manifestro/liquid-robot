from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from .model import CfCActorCritic, action_log_prob, action_entropy


def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        # next_values is zero for true terminal and V(terminal observation) for truncation.
        delta = rewards[t] + gamma * next_values[t] - values[t]
        last = delta + gamma * gae_lambda * (not dones[t]) * last
        advantages[t] = last
    return advantages, advantages + values


@dataclass
class Rollout:
    images: np.ndarray
    vectors: np.ndarray
    states: np.ndarray
    starts: np.ndarray
    z: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    next_values: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    @classmethod
    def allocate(cls, size: int, hidden_size: int):
        return cls(
            np.empty((size, 3, 120, 160), np.uint8),
            np.empty((size, 1118), np.float32),
            np.empty((size, hidden_size), np.float32),
            np.zeros(size, bool),
            np.empty((size, 2), np.float32),
            np.empty(size, np.float32),
            np.empty(size, np.float32),
            np.empty(size, np.float32),
            np.empty(size, np.float32),
            np.zeros(size, bool),
        )


def sequence_batch(rollout: Rollout, starts: list[int], length: int, burn_in: int, device):
    """Keep each chunk chronological; left-pad burn-in, right-pad the final short chunk."""
    batch, total = len(starts), burn_in + length
    indices = np.zeros((batch, total), np.int64)
    valid = np.zeros((batch, total), bool)
    initial = []
    for row, start in enumerate(starts):
        context = min(start, burn_in)
        prefix = start - context
        count = min(length, len(rollout.rewards) - start)
        offset = burn_in - context
        indices[row, offset : burn_in + count] = np.arange(prefix, start + count)
        valid[row, offset : burn_in + count] = True
        initial.append(rollout.states[prefix])
    mask = valid.copy()
    mask[:, :burn_in] = False
    fields = {
        "images": rollout.images[indices],
        "vectors": rollout.vectors[indices],
        "episode_starts": rollout.starts[indices],
        "z": rollout.z[indices],
        "old_log_probs": rollout.log_probs[indices],
        "old_values": rollout.values[indices],
        "advantages": rollout.advantages[indices],
        "returns": rollout.returns[indices],
        "state": np.stack(initial),
        "valid": valid,
        "mask": mask,
    }
    return {name: torch.as_tensor(value, device=device) for name, value in fields.items()}


def masked_mean(value, mask):
    return value[mask].mean()


def update_ppo(
    model: CfCActorCritic,
    optimizer,
    rollout: Rollout,
    *,
    epochs=4,
    sequence_length=32,
    burn_in=8,
    sequences_per_batch=4,
    clip=0.2,
    value_coef=0.5,
    entropy_coef=0.005,
    max_grad_norm=0.5,
    target_kl=0.03,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    device = next(model.parameters()).device
    advantages = rollout.advantages
    rollout.advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
    chunk_starts = np.arange(0, len(rollout.rewards), sequence_length)
    metrics = []
    early_stop = False
    for _ in range(epochs):
        order = rng.permutation(chunk_starts)
        for offset in range(0, len(order), sequences_per_batch):
            batch = sequence_batch(
                rollout, order[offset : offset + sequences_per_batch].tolist(), sequence_length, burn_in, device
            )
            mean, log_std, value, _ = model.forward_sequence(
                batch["images"],
                batch["vectors"],
                batch["state"],
                batch["episode_starts"],
                batch["valid"],
                burn_in=burn_in,
            )
            log_prob = action_log_prob(batch["z"], mean, log_std)
            log_ratio = log_prob - batch["old_log_probs"]
            ratio = log_ratio.exp()
            mask = batch["mask"]
            objective = ratio * batch["advantages"]
            clipped = ratio.clamp(1 - clip, 1 + clip) * batch["advantages"]
            policy_loss = -masked_mean(torch.minimum(objective, clipped), mask)
            clipped_value = batch["old_values"] + (value - batch["old_values"]).clamp(-clip, clip)
            value_loss = 0.5 * masked_mean(
                torch.maximum((value - batch["returns"]) ** 2, (clipped_value - batch["returns"]) ** 2), mask
            )
            entropy = masked_mean(action_entropy(mean, log_std), mask)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                model.log_std.clamp_(-3, 0.5)
                kl = masked_mean((ratio - 1) - log_ratio, mask)
            metrics.append(
                {
                    "loss": float(loss.detach()),
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "approx_kl": float(kl),
                    "clip_fraction": float(masked_mean((abs(ratio - 1) > clip).float(), mask)),
                    "grad_norm": float(grad_norm),
                }
            )
            if kl > target_kl:
                early_stop = True
                break
        if early_stop:
            break
    return {
        **{key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
        "optimizer_steps": len(metrics),
        "kl_early_stop": early_stop,
    }

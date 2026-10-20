"""Unity-independent, stateful deployment boundary for a trusted CfC checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from steppe_robot.contract import CAMERA_SHAPE, VECTOR_SIZE
from steppe_robot.rl.model import CfCActorCritic


class Policy:
    """One policy instance per robot/session. Call reset at episode/session boundaries.

    Inputs must already follow observation-v1 (including navigation and timestamps).
    This API emits normalized left/right commands; it does not drive hardware.
    Instances keep recurrent state and are not thread safe.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1 or payload.get("algorithm") != "CfC-sequence-PPO":
            raise ValueError("Unsupported checkpoint format or algorithm")
        self.device = torch.device(device)
        self.model = CfCActorCritic(payload["config"]["hidden_size"]).to(self.device).eval()
        self.model.load_state_dict(payload["model"], strict=True)
        if not all(torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("Checkpoint contains non-finite weights")
        self.training_step = int(payload["step"])
        self.curriculum = payload["config"].get("curriculum", "navigation")
        self.reset()

    def reset(self) -> None:
        self._state = self.model.initial_state(1, self.device)
        self._episode_start = True

    @torch.inference_mode()
    def act(self, image: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if image.shape != CAMERA_SHAPE or image.dtype != np.uint8:
            raise ValueError(f"Expected RGB uint8 CHW image {CAMERA_SHAPE}")
        if vector.shape != (VECTOR_SIZE,) or not np.issubdtype(vector.dtype, np.floating):
            raise ValueError(f"Expected floating observation vector ({VECTOR_SIZE},)")
        if not np.isfinite(vector).all():
            raise ValueError("Observation contains NaN or infinity")
        # Failures leave memory intact; commit the candidate state only after validation.
        action, _, _, _, next_state = self.model.act(
            np.ascontiguousarray(image),
            np.ascontiguousarray(vector, dtype=np.float32),
            self._state,
            self._episode_start,
            deterministic=True,
        )
        if action.shape != (2,) or not np.isfinite(action).all() or np.abs(action).max() > 1:
            raise RuntimeError("Policy produced an invalid drive command")
        self._state, self._episode_start = next_state, False
        return action.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded observations through a stateful CfC policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input", required=True, help="NPZ: images [N,3,120,160], vectors [N,1118], episode_starts [N]"
    )
    parser.add_argument("--output", required=True, help="New NPZ containing actions [N,2]")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    policy = Policy(args.checkpoint, args.device)
    with np.load(args.input, allow_pickle=False) as data:
        images, vectors, starts = data["images"], data["vectors"], data["episode_starts"]
        if len(images) == 0 or images.shape[0] != vectors.shape[0] or starts.shape != (len(images),):
            raise ValueError("Recorded arrays must have the same nonzero length")
        if starts.dtype != np.bool_:
            raise ValueError("episode_starts must be boolean")
        actions = []
        for image, vector, start in zip(images, vectors, starts):
            if start:
                policy.reset()
            actions.append(policy.act(image, vector))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, actions=np.stack(actions))


if __name__ == "__main__":
    main()

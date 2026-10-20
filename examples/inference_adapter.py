"""Integration boundary: bring synchronized, contract-compliant sensor observations."""

from pathlib import Path

import numpy as np

from steppe_robot.inference import Policy


class RobotSession:
    def __init__(self, checkpoint: Path):
        self.policy = Policy(checkpoint, device="cpu")

    def begin_journey(self) -> None:
        self.policy.reset()

    def decide(self, rgb_chw: np.ndarray, observation_v1: np.ndarray) -> tuple[float, float]:
        left, right = self.policy.act(rgb_chw, observation_v1)
        return float(left), float(right)

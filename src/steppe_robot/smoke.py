"""Exercise LLAPI only: no training, Torch, model weights or recurrent state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from steppe_robot.contract import BEHAVIOR_PREFIX, CAMERA_SHAPE, VECTOR_SIZE, FIELDS, validate_observation
from steppe_robot.messages import EpisodeMessages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="../steppe-unity/Builds/RobotTraining/RobotTraining.exe")
    parser.add_argument("--editor", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--report", default="runs/bridge-smoke.json")
    args = parser.parse_args()
    engine, parameters, messages = EngineConfigurationChannel(), EnvironmentParametersChannel(), EpisodeMessages()
    engine.set_configuration_parameters(width=960, height=720, quality_level=1, time_scale=4)
    parameters.set_float_parameter("master_seed", 12345)
    parameters.set_float_parameter("evaluation", 0)
    parameters.set_float_parameter("world_seed", -1)
    parameters.set_float_parameter("timeout_seconds", 90)
    env = UnityEnvironment(
        file_name=None if args.editor else str(Path(args.env).resolve()),
        worker_id=args.worker_id,
        side_channels=[engine, parameters, messages],
        no_graphics=False,
        timeout_wait=180,
        log_folder="runs/unity-player",
    )
    report = {}
    try:
        env.reset()
        behavior = next(name for name in env.behavior_specs if name.startswith(BEHAVIOR_PREFIX))
        spec = env.behavior_specs[behavior]
        assert [o.shape for o in spec.observation_specs] == [CAMERA_SHAPE, (VECTOR_SIZE,)]
        assert spec.action_spec.continuous_size == 2 and spec.action_spec.discrete_size == 0

        def decision():
            steps, terminal = env.get_steps(behavior)
            assert len(terminal) == 0, "Unexpected early termination"
            assert len(steps) == 1
            validate_observation(steps.obs[0][0], steps.obs[1][0])
            return steps.obs[1][0].copy(), steps.obs[0][0]

        first, image = decision()
        assert image.std() > 0.001, "Camera is blank"
        starts = [m for m in messages.messages if m["type"] == "episode_start"]
        seed = starts[-1]["seed"]
        assert seed % 2 == 0
        for _ in range(20):
            env.set_actions(behavior, ActionTuple(continuous=np.array([[0.6, 0.6]], dtype=np.float32)))
            env.step()
            vector, _ = decision()
            assert abs(float(vector[1095]) - 0.1) < 1e-4, vector[1095]
            assert vector[1096:1100].max() <= 0.1001
        travel = float(np.linalg.norm(vector[FIELDS["gps_position_m"]] - first[FIELDS["gps_position_m"]]))
        assert travel > 0.2, ("Actions did not move the robot", travel)
        for _ in range(20):
            env.set_actions(behavior, ActionTuple(continuous=np.zeros((1, 2), dtype=np.float32)))
            env.step()
            vector, _ = decision()
        assert np.abs(vector[FIELDS["wheel_rpm"]]).max() < 1
        assert np.linalg.norm(vector[FIELDS["gps_velocity_mps"]]) < 0.05

        # Replay the exact seed, including goal, spawn, and fresh sensor history.
        parameters.set_float_parameter("world_seed", seed)
        env.reset()
        replay, _ = decision()
        np.testing.assert_allclose(replay[FIELDS["gps_position_m"]], first[FIELDS["gps_position_m"]], atol=0.005)
        np.testing.assert_allclose(replay[1090:1093], first[1090:1093], atol=0.005)
        assert abs(float(replay[1095])) < 1e-6
        assert np.abs(replay[FIELDS["wheel_rpm"]]).max() < 1

        parameters.set_float_parameter("lidar_failure", 1)
        env.reset()
        fault, _ = decision()
        assert (fault[FIELDS["lidar_error"]] == 1).all()
        assert (fault[FIELDS["lidar_no_hit"]] == 0).all()
        assert (fault[FIELDS["lidar_m"]] == 0).all()
        parameters.set_float_parameter("lidar_failure", 0)

        # Hold-out seeds are odd; timeout must arrive as truncation (interrupted=True).
        parameters.set_float_parameter("world_seed", -1)
        parameters.set_float_parameter("evaluation", 1)
        parameters.set_float_parameter("timeout_seconds", 0.4)
        env.reset()
        decision()
        eval_seed = [m for m in messages.messages if m["type"] == "episode_start"][-1]["seed"]
        assert eval_seed % 2 == 1
        terminal_seen = False
        for _ in range(12):
            steps, terminal = env.get_steps(behavior)
            if len(terminal):
                assert terminal.interrupted.all()
                terminal_seen = True
                break
            if len(steps):
                env.set_actions(behavior, ActionTuple(continuous=np.zeros((len(steps), 2), dtype=np.float32)))
            env.step()
        assert terminal_seen, "Timeout did not finish the episode"
        assert any(m["type"] == "episode" and m["reason"] == "timeout" for m in messages.messages)
        report = {
            "passed": True,
            "behavior": behavior,
            "camera_shape": CAMERA_SHAPE,
            "vector_size": VECTOR_SIZE,
            "train_seed": seed,
            "evaluation_seed": eval_seed,
            "travel_m": travel,
            "checks": [
                "RGB",
                "lidar_masks",
                "sensor_ages",
                "10Hz_actions",
                "drive",
                "brake",
                "seed_replay",
                "sensor_reset",
                "lidar_fault",
                "holdout_split",
                "timeout_terminal",
            ],
        }
        print(json.dumps(report, indent=2))
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from steppe_robot.contract import BEHAVIOR_PREFIX, CAMERA_SHAPE, VECTOR_SIZE, validate_observation
from steppe_robot.messages import EpisodeMessages


@dataclass
class Observation:
    image: np.ndarray  # Quantized RGB, identical at collection and PPO replay.
    vector: np.ndarray


@dataclass
class Transition:
    observation: Observation
    reward: float
    done: bool
    interrupted: bool
    terminal_observation: Observation | None
    metrics: list[dict]


class UnityRobotEnv:
    def __init__(
        self,
        executable,
        *,
        worker_id=10,
        seed=42,
        time_scale=20,
        log_folder="runs/unity-player",
        curriculum="navigation",
        background=False,
    ):
        self.curriculum = curriculum
        engine = EngineConfigurationChannel()
        engine.set_configuration_parameters(width=640, height=480, quality_level=0, time_scale=time_scale)
        self.parameters = EnvironmentParametersChannel()
        self.messages = EpisodeMessages()
        self.parameters.set_float_parameter("master_seed", seed)
        self.parameters.set_float_parameter("evaluation", 0)
        self.parameters.set_float_parameter("world_seed", -1)
        self.parameters.set_float_parameter("timeout_seconds", 90)
        if curriculum == "straight":
            self.parameters.set_float_parameter("goal_min_distance", 3)
            self.parameters.set_float_parameter("goal_max_distance", 5)
            self.parameters.set_float_parameter("straight_goal", 1)
        elif curriculum == "gps_long":
            self.parameters.set_float_parameter("goal_min_distance", 10)
            self.parameters.set_float_parameter("goal_max_distance", 30)
            self.parameters.set_float_parameter("gps_navigation", 1)
            self.parameters.set_float_parameter("straight_goal", 0)
        self.env = UnityEnvironment(
            file_name=str(Path(executable).resolve()),
            seed=seed,
            worker_id=worker_id,
            side_channels=[engine, self.parameters, self.messages],
            no_graphics=False,
            timeout_wait=180,
            log_folder=str(log_folder),
            additional_args=["-batchmode"] if background else None,
        )
        self.behavior = None
        self.agent_id = None
        self.message_cursor = 0

    def _observe(self, steps, index=0):
        image, vector = steps.obs[0][index], steps.obs[1][index]
        validate_observation(image, vector)
        return Observation(np.rint(image * 255).astype(np.uint8), vector.copy())

    def reset(self, *, evaluation=False, world_seed=-1, timeout=90):
        self.parameters.set_float_parameter("evaluation", float(evaluation))
        self.parameters.set_float_parameter("world_seed", world_seed)
        self.parameters.set_float_parameter("timeout_seconds", timeout)
        self.env.reset()
        self.behavior = next(name for name in self.env.behavior_specs if name.startswith(BEHAVIOR_PREFIX))
        spec = self.env.behavior_specs[self.behavior]
        if [o.shape for o in spec.observation_specs] != [
            CAMERA_SHAPE,
            (VECTOR_SIZE,),
        ] or spec.action_spec.continuous_size != 2:
            raise RuntimeError("Unity observation/action contract mismatch")
        self.message_cursor = len(self.messages.messages)
        observation = self._next_decision()
        if observation.image.std() < 0.25:
            raise RuntimeError("Unity camera is blank")
        if self.curriculum == "straight":
            if not (
                2.8 <= observation.vector[1090] <= 5.2
                and abs(observation.vector[1091]) < 0.03
                and observation.vector[1092] > 0.99
            ):
                raise RuntimeError("Stage 1 requires a goal 3-5 m straight ahead; rebuild the Unity environment")
        if self.curriculum == "gps_long":
            start = next((m for m in reversed(self.messages.messages) if m.get("type") == "episode_start"), {})
            if not start.get("gps_navigation") or not 9.5 <= observation.vector[1090] <= 32:
                raise RuntimeError("GPS stage requires the new GPS-navigation build and a 10-30 m goal")
        return observation

    def _next_decision(self):
        for _ in range(100):
            decision, terminal = self.env.get_steps(self.behavior)
            if len(decision):
                if len(decision) != 1:
                    raise RuntimeError("This experiment supports exactly one robot")
                self.agent_id = int(decision.agent_id[0])
                return self._observe(decision)
            self.env.step()  # No actions while the new episode settles.
        raise RuntimeError("Unity failed to produce the next decision")

    def step(self, action: np.ndarray):
        if not np.isfinite(action).all() or action.shape != (2,) or np.abs(action).max() > 1.00001:
            raise ValueError("Expected two finite actions in [-1,1]")
        self.env.set_actions(self.behavior, ActionTuple(continuous=action.astype(np.float32)[None]))
        old_id = self.agent_id
        self.env.step()
        decision, terminal = self.env.get_steps(self.behavior)
        done = old_id in terminal.agent_id_to_index
        if done:
            row = terminal.agent_id_to_index[old_id]
            reward, interrupted = float(terminal.reward[row]), bool(terminal.interrupted[row])
            final = self._observe(terminal, row)
            observation = self._next_decision()
        else:
            if old_id not in decision.agent_id_to_index:
                raise RuntimeError("Missing decision for the current robot")
            row = decision.agent_id_to_index[old_id]
            reward, interrupted, final = float(decision.reward[row]), False, None
            observation = self._observe(decision, row)
        messages = self.messages.messages[self.message_cursor :]
        self.message_cursor = len(self.messages.messages)
        metrics = [m for m in messages if m.get("type") == "episode"]
        return Transition(observation, reward, done, interrupted, final, metrics)

    def close(self):
        self.env.close()

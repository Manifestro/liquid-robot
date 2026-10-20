"""Version 1 Unity/Python contract. All scalars are raw SI values except RPM/degrees."""

from __future__ import annotations

import numpy as np

BEHAVIOR_PREFIX = "SteppeRobot-v1"
CAMERA_SHAPE = (3, 120, 160)  # CHW in ML-Agents 4.1; float32 RGB in [0, 1].
VECTOR_SIZE = 1118
FIELDS = {
    "lidar_m": slice(0, 360),
    "lidar_no_hit": slice(360, 720),
    "lidar_error": slice(720, 1080),
    "imu_accel_mps2": slice(1080, 1083),
    "imu_gyro_radps": slice(1083, 1086),
    "wheel_rpm": slice(1086, 1090),
    "goal_distance_m": 1090,
    "goal_right": 1091,
    "goal_forward": 1092,
    "simulation_seconds": 1093,
    "episode_seconds": 1094,
    "observation_dt_seconds": 1095,
    "camera_age_seconds": 1096,
    "lidar_age_seconds": 1097,
    "imu_age_seconds": 1098,
    "wheel_age_seconds": 1099,
    "goal_age_seconds": 1100,
    "camera_valid": 1101,
    "lidar_scan_valid": 1102,
    "imu_valid": 1103,
    "wheel_valid": 1104,
    "gps_age_seconds": 1105,
    "gps_position_m": slice(1106, 1109),
    "gps_velocity_mps": slice(1109, 1112),
    "gps_course_degrees": 1112,
    "gps_course_valid": 1113,
    "gps_valid": 1114,
    "goal_valid": 1115,
    "physics_dt_seconds": 1116,
    "action_dt_seconds": 1117,
}


def validate_observation(camera: np.ndarray, vector: np.ndarray) -> None:
    if camera.shape != CAMERA_SHAPE or vector.shape != (VECTOR_SIZE,):
        raise ValueError(f"Observation shape mismatch: {camera.shape}, {vector.shape}")
    if not np.isfinite(camera).all() or not np.isfinite(vector).all():
        raise ValueError("Non-finite observation")
    if not 0 <= camera.min() <= camera.max() <= 1:
        raise ValueError("Wire RGB must be normalized to [0,1]")
    distance, no_hit, error = (vector[FIELDS[name]] for name in ("lidar_m", "lidar_no_hit", "lidar_error"))
    if not (
        np.all((0 <= distance) & (distance <= 12))
        and np.isin(no_hit, [0, 1]).all()
        and np.isin(error, [0, 1]).all()
        and np.all(no_hit + error <= 1)
        and np.all(distance[no_hit == 1] == 12)
        and np.all(distance[error == 1] == 0)
    ):
        raise ValueError("Invalid lidar range or masks")
    if not np.all(vector[1096:1101] >= 0):
        raise ValueError("Negative sensor age")
    if not (
        np.isclose(vector[FIELDS["physics_dt_seconds"]], 0.02) and np.isclose(vector[FIELDS["action_dt_seconds"]], 0.1)
    ):
        raise ValueError("Unexpected physics/action interval")

# Protocol v1

The authoritative machine-readable layout is `contracts/observation-v1.json`. Both repositories carry byte-identical snapshots. Compare SHA-256 when pairing releases. Protocol v1 has behavior prefix `SteppeRobot-v1`, vector length 1118, CHW RGB 3×120×160 and two continuous actions in [-1,1] ordered left/right. Physics is 50 Hz; decisions are 10 Hz.

Unity sends normalized float32 RGB; Python quantizes once to uint8 and uses the same pixels for rollout/replay. Distances use metres, acceleration m/s², angular velocity rad/s, wheel speed rpm, course degrees. Coordinates: +X right, +Y up, +Z forward. Course 0° is +Z, increasing toward +X. Lidar ray zero is forward, increasing clockwise viewed from above. Missing hit uses 12 m and no_hit=1; sensor errors use distance=0 and error=1.

GPS is local metric position/velocity. At low speed course is invalid. IMU acceleration is specific force (a − gravity) in sensor axes. Timestamps are simulation seconds, not host wall time. Sensor ages, validity masks, measured dt and nominal rates are separate fields. Wheels: left-front, left-rear, right-front, right-rear.

Episode side-channel UUID: `9c449eb4-0a7b-4d32-87ad-6bbb0b236251`. Messages: schema, episode_start and episode. Terminals include success, collision, overturn, out_of_bounds, timeout. Timeout is interrupted and bootstraps the terminal observation; all terminals reset policy memory. World geometry changes only on episode reset. Even seeds train, odd seeds evaluate; explicit replay seeds are recorded.

Actions request target side rpm, subject to motor torque/slew/braking and physics contact. They do not set pose. The two left motors share one command, the two right motors another. Zero command brakes.

Semantic changes also matter: `straight`/`navigation` use ideal relative goal direction; `gps_long` uses GPS position and GPS/gyro estimated heading, with a known north-facing launch. Record curriculum in checkpoint/run manifests. GPS heading during sliding is course over ground, not necessarily body yaw; current sensors have no noise/delay. Initial orientation estimation and calibrated real sensor adapters remain future work.

An incompatible shape, unit, axis or action change requires protocol v2, a new behavior prefix and coordinated release tests. Do not silently reuse v1 checkpoints. Algorithm changes with identical I/O still require evaluation against fixed held-out seeds and recorded simulator build hashes.

# Architecture

Unity produces physics and timestamped observations. `rl.unity_env.UnityRobotEnv` validates LLAPI shapes and packages uint8 camera frames with float vectors. The training loop collects chronological rollouts, constructs correct terminal bootstraps, updates recurrent PPO, refreshes ongoing episode memory and writes atomic checkpoints and status files. Evaluation uses deterministic actions and separate seeds.

The model encodes RGB with a CNN, lidar with circular 1D convolutions and proprioception/relative goal/ages with an MLP, then fuses them into a CfC cell (128 hidden units). Separate actor and critic heads produce a tanh-Gaussian policy and value. Absolute GPS coordinates and absolute clocks are excluded from policy features. There are 246789 parameters in the default configuration.

PPO uses rollout 256, sequence 32, burn-in 8, four epochs, masks for padding/reset, correct tanh Jacobian, GAE with separate timeout bootstrap and bounded gradients. Episode memory resets only at boundaries; after weight updates ongoing memory is reconstructed from observation history. See tests and TRAINING.md for details and truncation limits.

`inference.Policy` depends on NumPy, PyTorch and ncps only. It owns one robot's recurrent state and validates the input/output boundary. It never connects to Unity or motors. A host adapter must synchronize sensors, compute relative GPS goals/estimated course, build observation-v1, schedule 10 Hz decisions and enforce motor constraints/stops independently. Each session gets its own Policy instance.

The contract snapshot is duplicated deliberately to keep both repositories independently clonable. It changes only through a coordinated versioned release. Checkpoint format 1 contains architecture config, model, optimizer and RNG metadata; weights-only loading is used. Package distribution contains source only, not trained weights.

# Validation record — 2026-10-20

## Follow-up — 2026-10-21, version 0.2.0

24 tests passed, including ranking/regression protection and validation/test seed isolation; Ruff lint/format checks passed. A real Unity integration test ran 64 training actions with an initial and final validation in a second player, retained the initial weights when the updated policy scored worse, and evaluated selected-best weights on a separate test seed. Both Unity processes closed cleanly. This bounded run tested the selection mechanism, not new navigation quality.

A separate frozen-checkpoint benchmark completed 108 episodes: four models on 12 selection worlds each, then the selected step 100352 and final step 200000 on 30 disjoint confirmation worlds. Results were 30/30 versus 21/30 successful routes, with 0 versus 4 collisions. See [experiment record](experiments/gps-checkpoints-2026-10-21.md). Checkpoints were not updated during benchmarking; the selected copy is verified against its original SHA-256. Final-test outcomes did not choose the checkpoint.

The original extraction validation follows below.

Validated on Windows, Python 3.10.12, Torch 2.8.0+cu126, RTX 3060. A separate package-test environment reused the existing runtime dependency installation through a local `.pth`; it did not modify the active trainer's environment. The `.pth` and virtual environments are ignored local artifacts, not an installation requirement.

| Check | Result |
|---|---|
| Unit/integration tests | 15 passed: recurrent forward/reset, padding, GAE/timeouts, tanh log probability, PPO weight update, memory refresh, collector terminals, stateful inference/reset/error handling, CLI replay, protocol and upstream provenance |
| Static checks | Ruff lint and format checks passed for project source, tests and example |
| Package build | Source distribution and universal Python wheel built successfully |
| Installed wheel | Imported outside the checkout in an isolated wheel-test environment; loaded the existing 100000-step checkpoint, ran deterministic inference and loaded the packaged JSON schema |
| No Unity import for inference | Verified in a fresh subprocess |
| Real Unity LLAPI | Installed `steppe-smoke` passed RGB/lidar/masks/ages, 10 Hz actions, driving/braking, seed replay, sensor reset, lidar failure, holdout split and timeout; travel 1.495 m |
| Extracted repo pair | Installed `steppe-train` ran 64 actions against the newly built companion player in GPS mode; weights changed and latest checkpoint was saved |
| Dependency resolution | `uv.lock`: 64 resolved packages, hashes and platform metadata; compatible ML-Agents source included with pinned commit and provenance hashes |

Commands and test implementation are in README and `tests/`. Local smoke outputs/build logs live under ignored `runs/`; distributions under ignored `dist/`. No model weights or experimental data are included in Git. The running long-distance experiment remains in the original workspace and is not a run of this extracted package.

The dependency lock was resolved locally; a fresh download/install of every platform wheel was not performed. The installed-wheel test reused runtime dependencies, while CI is configured to install its own CPU dependencies. Remote CI has not run because no remote repository is configured yet. These checks establish implementation behavior, not real-robot readiness or broad navigation success.

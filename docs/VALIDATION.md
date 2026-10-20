# Validation record — 2026-10-20

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

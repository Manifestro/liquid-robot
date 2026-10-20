# Steppe Robot Learning

PyTorch CfC actor–critic with recurrent PPO, Unity LLAPI integration, evaluation tools and a Unity-independent stateful inference API. Companion simulator: **steppe-unity**. Python **3.10.12**, protocol **v1**, Unity ML-Agents **4.1.0** (Python package from the pinned matching upstream commit).

## Install

Clone the two repositories into sibling directories. Run from this repository:

```powershell
uv python install 3.10.12
uv venv --python 3.10.12
uv pip install ./vendor/mlagents-envs
uv pip install -e ".[unity,training,dev]"
.venv/Scripts/python.exe -m pytest
```

For CPU-only Torch, install `torch==2.8.0` from `https://download.pytorch.org/whl/cpu` first. For the validated NVIDIA setup, install `torch==2.8.0` from `https://download.pytorch.org/whl/cu126` first, then install this package. The package does not force a CUDA build on deployment machines. Linux entry points are under `.venv/bin/`.

Inference only: `uv pip install .` (Unity, ML-Agents and plotting are optional). See [inference](docs/INFERENCE.md).

The compatible upstream `mlagents-envs` development version is supplied as pristine source in `vendor/` so installation does not clone Unity's large Git history. Its commit, license and file hashes are recorded in [THIRD_PARTY.md](THIRD_PARTY.md). For the fully locked default environment use `uv sync --locked --extra unity --extra training --extra dev`; platform-specific Torch wheels are selected by uv. The explicit pip commands above also support choosing a CPU/CUDA index first.

## Train and evaluate

First build the companion Unity project through **Steppe → Build Python Environment**. Pass its executable explicitly:

```powershell
.venv/Scripts/steppe-train.exe --env ../steppe-unity/Builds/RobotTraining/RobotTraining.exe --run runs/straight --curriculum straight --timeout 20 --steps 100000
.venv/Scripts/steppe-train.exe --env ../steppe-unity/Builds/RobotTraining/RobotTraining.exe --run runs/gps-long --curriculum gps_long --timeout 90 --steps 200000 --initialize-from runs/straight/latest.pt
.venv/Scripts/steppe-evaluate.exe --env ../steppe-unity/Builds/RobotTraining/RobotTraining.exe --checkpoint runs/gps-long/latest.pt --out runs/gps-watch --seeds 123 1001 2027
.venv/Scripts/steppe-report.exe --run runs/gps-long
```

Each run requires a **new directory**. `--initialize-from` transfers weights and starts a new optimizer/counter; it is not exact simulator resume. For CPU training pass `--device cpu`. RGB rendering requires a graphics device; do not pass Unity `-nographics`. Use unique `--worker-id` values for concurrent environments.

Monitor in two terminals:

```powershell
.venv/Scripts/steppe-monitor.exe --run runs/gps-long
.venv/Scripts/python.exe -m tensorboard.main --logdir runs/gps-long/tensorboard --host 127.0.0.1 --port 6006 --load_fast=false
```

## Repository map

| Path | Responsibility |
|---|---|
| `src/steppe_robot/rl/` | Model, recurrent PPO, Unity environment adapter |
| `src/steppe_robot/inference.py` | Stateful deterministic inference, no Unity dependency |
| `src/steppe_robot/training.py` | Bounded experiments, evaluation, checkpoints and manifests |
| `src/steppe_robot/contract.py`, `contracts/` | Observation layout and shared protocol snapshot |
| `tests/` | Sequence correctness, PPO/GAE, inference, protocol and CLI tests |
| `examples/` | Hardware adapter boundary example |
| `docs/` | Architecture, setup, experiment and deployment notes |

## Validation

```powershell
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m ruff check src tests examples
.venv/Scripts/python.exe -m ruff format --check src tests examples
.venv/Scripts/python.exe -m build
```

CI runs these on CPU; Unity integration needs a licensed simulator build and is tested separately. No run directories or weights are committed. See [architecture](docs/ARCHITECTURE.md), [training](docs/TRAINING.md), [protocol](docs/PROTOCOL.md), [migration](docs/MIGRATION.md) and [validation record](docs/VALIDATION.md).

`uv.lock` records the complete default dependency resolution. When updating dependencies, regenerate it with `uv lock`, rerun validation and review the diff. CPU/CUDA wheel overrides using the explicit pip install route should be recorded with each run's environment manifest.

This is a simulation research package. Geometric dimensions reflect supplied measurements; motor/tire parameters and sensor errors still need calibration. No hardware driver or real-robot deployment is included. No project redistribution license has been selected; upstream dependencies retain their own licenses.

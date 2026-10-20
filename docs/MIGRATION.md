# Extraction and artifact ownership

These repositories were prepared as independent copies while an experiment was running in the original workspace. That workspace, its virtual environment, Unity caches, player builds, model weights and experiment logs remain intact. No running process was redirected to these copies.

Place `steppe-unity` and `steppe-robot-learning` beside each other, clone/install normally, and build a fresh simulator. Always pass its executable with `--env`. Run outputs belong under the Python repository's ignored `runs/`; Unity writes player-side episode logs beside its executable. Existing `.pt` format-v1 weights remain loadable by the Python package because parameter names and architecture are unchanged. For cross-stage initialization use `--initialize-from`; simulator and optimizer state do not resume exactly.

Existing local artifacts can be supplied by absolute path; they are not automatically copied or committed. Preserve manifests and source/build hashes together with weights. Training checkpoints, datasets and large binaries belong in separately managed artifact storage or release assets once that storage is selected. Remote URLs and permissions will be provided by the owner; neither repository currently has a configured remote.

Local launch supervisors from the original machine are intentionally not part of the package. Use portable CLI entry points; a scheduler/process manager may supervise them on the deployment host. No Codex-specific timers are embedded in either repository.

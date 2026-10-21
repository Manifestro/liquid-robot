# Changelog

## 0.2.0

- Compare frozen checkpoints on separate selection and confirmation worlds; save selected weights and trajectory reports.
- Periodically validate training in a separate Unity process and retain `best.pt` independently of `latest.pt`.
- Evaluate both latest and validation-selected weights on a separate final test set (30 default worlds).
- Expose learning rate, validation interval and validation/test seeds through CLI; reject overlapping seed sets.
- Reproduce and document the late-training regression in GPS navigation without modifying existing checkpoints.

## 0.1.0

- Initial standalone repository, extracted without changing the active experiment.
- Observation protocol v1; compatible Unity/Python integration.
- Existing runtime caches, checkpoints and experiment logs remain outside source control.

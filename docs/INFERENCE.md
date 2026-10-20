# Stateful inference

Install the base package on Python 3.10.12. Unity/ML-Agents is not required. Use a trusted format-v1 checkpoint:

```python
from steppe_robot.inference import Policy

policy = Policy('/absolute/path/latest.pt', device='cpu')
policy.reset()  # once per new episode/session
# image: numpy uint8 [3,120,160], RGB, CHW
# vector: numpy float32 [1118], following contracts/observation-v1.json
left, right = policy.act(image, vector)
```

Keep memory across decisions and reset it when starting a new journey, reconnecting after a discontinuity or resetting the simulator. One instance belongs to one robot/session; it is not thread safe. Decisions are deterministic. Do not reset memory at every frame. Input validation failures do not advance recurrent state.

Offline replay CLI:

```powershell
steppe-predict --checkpoint artifacts/latest.pt --input observations.npz --output actions.npz --device cpu
```

Input NPZ arrays: images `[N,3,120,160]` uint8, vectors `[N,1118]` floating, episode_starts `[N]` boolean. Output: actions `[N,2]`. Output files are created exclusively to prevent accidental replacement.

The hardware adapter is a deployment boundary, not an implemented driver: see `examples/inference_adapter.py`. It must supply consistent units/axes, camera resizing, lidar masks, sensor timestamps/validity, goal position, heading estimation and normalized motor commands. Simulator GPS is local metres; geographic GPS needs a local tangent-plane conversion. GPS heading is course over ground; stationary heading cannot be obtained from it. MPU6050 has no magnetometer. The present GPS curriculum assumes known north-facing launch and has not been calibrated against real sensor noise or delay.

Deploy with independently enforced stale-data stops, command limits and an emergency stop. Current validation proves software behavior in simulation, not real-world autonomous operation. TorchScript/ONNX export and hardware transport are not implemented or claimed. Checkpoints/robot logs are not bundled with the package.

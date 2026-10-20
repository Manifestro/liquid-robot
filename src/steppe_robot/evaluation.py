"""Watch a saved deterministic CfC policy drive the Unity robot."""

import argparse
from pathlib import Path
import json
import torch
from steppe_robot.rl.model import CfCActorCritic
from steppe_robot.rl.unity_env import UnityRobotEnv
from steppe_robot.training import Config, evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/latest.pt")
    parser.add_argument("--env", default=Config.env)
    parser.add_argument("--out", default="runs/cfc_policy_watch")
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 1001, 2027])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--worker-id", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"])
    config.env, config.device, config.time_scale = args.env, args.device, args.time_scale
    config.eval_seeds = tuple(args.seeds)
    if args.timeout is not None:
        config.timeout = max(0.2, args.timeout)
    model = CfCActorCritic(config.hidden_size).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    env = UnityRobotEnv(
        args.env,
        worker_id=args.worker_id,
        seed=config.seed,
        time_scale=args.time_scale,
        log_folder=out / "unity-player",
        curriculum=config.curriculum,
    )
    try:
        summary = evaluate(model, env, config, out, "policy")
        summary["checkpoint_step"] = checkpoint["step"]
        write_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()

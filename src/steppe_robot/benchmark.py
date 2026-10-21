"""Compare frozen checkpoints, select on one seed set, confirm on a disjoint set."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from steppe_robot.rl.model import CfCActorCritic
from steppe_robot.rl.unity_env import UnityRobotEnv
from steppe_robot.training import Config, evaluate, write_json


def summarize(rows):
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_episode_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])),
        "results": rows,
    }


def ranking(summary):
    return (summary["success_rate"], -summary["collision_rate"], summary["mean_return"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--confirmation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--worker-id", type=int, default=31)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    selection, confirmation = args.selection_seeds, args.confirmation_seeds
    if len(set(selection)) != len(selection) or len(set(confirmation)) != len(confirmation):
        parser.error("Seed sets must contain unique seeds")
    if set(selection) & set(confirmation) or any(seed % 2 != 1 for seed in selection + confirmation):
        parser.error("Selection/confirmation must be disjoint odd-seed sets")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    checkpoints = []
    for path in args.checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["config"]["curriculum"] != "gps_long":
            raise ValueError("Compare only checkpoints from the same GPS curriculum")
        checkpoints.append((Path(path), payload))
    labels = [str(payload["step"]) for _, payload in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError("Checkpoint steps must be unique")
    write_json(
        out / "manifest.json",
        {
            "checkpoints": [
                {"path": str(p.resolve()), "step": c["step"], "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p, c in checkpoints
            ],
            "selection_seeds": selection,
            "confirmation_seeds": confirmation,
            "selection_rule": "highest success, then fewest collisions, then highest mean return",
            "timeout_seconds": 90,
            "curriculum": "gps_long",
            "training_performed": False,
            "unity_assembly_sha256": hashlib.sha256(
                (Path(args.env).parent / "RobotTraining_Data/Managed/Assembly-CSharp.dll").read_bytes()
            ).hexdigest(),
        },
    )
    env = None
    start, completed = time.monotonic(), 0
    try:
        env = UnityRobotEnv(
            args.env,
            worker_id=args.worker_id,
            time_scale=30,
            log_folder=out / "unity-player",
            curriculum="gps_long",
            background=True,
        )

        def run(payload, seeds, phase):
            nonlocal completed
            model = CfCActorCritic(payload["config"]["hidden_size"]).to(args.device).eval()
            model.load_state_dict(payload["model"], strict=True)
            config = replace(Config(**payload["config"]), device=args.device, timeout=90)
            directory = out / phase / str(payload["step"])
            directory.mkdir(parents=True)
            rows = []
            for seed in seeds:
                write_json(
                    out / "status.json",
                    {
                        "status": "running",
                        "phase": phase,
                        "checkpoint_step": payload["step"],
                        "seed": seed,
                        "completed_episodes": completed,
                        "elapsed_seconds": time.monotonic() - start,
                    },
                )
                result = evaluate(model, env, replace(config, eval_seeds=(seed,)), directory, "policy")
                rows.extend(result["results"])
                completed += 1
                write_json(directory / "summary.json", summarize(rows))
            return summarize(rows)

        selected = {}
        for _, payload in checkpoints:
            selected[str(payload["step"])] = run(payload, selection, "selection")
            write_json(out / "selection.json", selected)
        winner = max(selected, key=lambda label: ranking(selected[label]))
        final = str(max(payload["step"] for _, payload in checkpoints))
        confirmed = {}
        for label in dict.fromkeys([winner, final]):
            payload = next(c for _, c in checkpoints if str(c["step"]) == label)
            confirmed[label] = run(payload, confirmation, "confirmation")
            write_json(out / "confirmation.json", confirmed)
        write_json(
            out / "summary.json",
            {
                "winner_step": int(winner),
                "final_step": int(final),
                "selection": selected,
                "confirmation": confirmed,
                "wall_seconds": time.monotonic() - start,
            },
        )
        write_json(
            out / "status.json",
            {
                "status": "completed",
                "winner_step": int(winner),
                "completed_episodes": completed,
                "wall_seconds": time.monotonic() - start,
            },
        )
    except BaseException as error:
        write_json(out / "failure.json", {"error": repr(error), "completed_episodes": completed})
        write_json(out / "status.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

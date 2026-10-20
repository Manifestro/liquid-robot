"""Read live JSONL logs and publish TensorBoard events without touching the trainer."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A concurrent append may leave the last line temporarily incomplete.
    return result


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/experiment")
    args = parser.parse_args()
    run = Path(args.run)
    writer = SummaryWriter(str(run / "tensorboard"), flush_secs=3)
    seen_updates, seen_episodes = 0, 0
    try:
        while True:
            updates = records(run / "updates.jsonl")
            episodes = records(run / "episodes.jsonl")
            status = read_json(run / "status.json")
            for row in updates[seen_updates:]:
                step = row["step"]
                relevant = [e for e in episodes if e["training_step"] <= step]
                target = status.get("target_steps", 100000)
                speed = row["decisions_per_second"]
                scalars = {
                    "01_Progress/steps": step,
                    "01_Progress/percent": 100 * step / target,
                    "01_Progress/remaining_minutes": (target - step) / max(speed, 0.01) / 60,
                    "02_Learning/success_percent_all": 100
                    * sum(e["success"] for e in relevant)
                    / max(len(relevant), 1),
                    "02_Learning/success_percent_last20": 100
                    * sum(e["success"] for e in relevant[-20:])
                    / max(len(relevant[-20:]), 1),
                    "02_Learning/collision_percent_last20": 100
                    * sum(e["collision"] for e in relevant[-20:])
                    / max(len(relevant[-20:]), 1),
                    "02_Learning/mean_return_last20": sum(e["reward"] for e in relevant[-20:])
                    / max(len(relevant[-20:]), 1),
                    "02_Learning/completed_episodes": len(relevant),
                    "02_Learning/goal_reached_count": sum(e["success"] for e in relevant),
                    "03_Optimization/policy_loss": row["policy_loss"],
                    "03_Optimization/value_loss": row["value_loss"],
                    "03_Optimization/entropy": row["entropy"],
                    "03_Optimization/approx_kl": row["approx_kl"],
                    "03_Optimization/grad_norm": row["grad_norm"],
                    "04_Performance/decisions_per_second": speed,
                }
                for tag, value in scalars.items():
                    writer.add_scalar(tag, value, step)
            for index, episode in enumerate(episodes[seen_episodes:], start=seen_episodes + 1):
                writer.add_scalar("05_Episodes/return", episode["reward"], index)
                writer.add_scalar("05_Episodes/duration_seconds", episode["elapsed_seconds"], index)
                writer.add_scalar("05_Episodes/success", float(episode["success"]), index)
            seen_updates, seen_episodes = len(updates), len(episodes)
            if updates:
                latest = updates[-1]
                last20 = episodes[-20:]
                snapshot = {
                    "status": status.get("status", "unknown"),
                    "step": latest["step"],
                    "target_steps": status.get("target_steps", 100000),
                    "percent": latest["step"] / status.get("target_steps", 100000) * 100,
                    "speed": latest["decisions_per_second"],
                    "remaining_minutes": max(0, status.get("target_steps", 100000) - latest["step"])
                    / latest["decisions_per_second"]
                    / 60,
                    "episodes": len(episodes),
                    "successes": sum(e["success"] for e in episodes),
                    "success_percent_all": 100 * sum(e["success"] for e in episodes) / max(len(episodes), 1),
                    "success_percent_last20": 100 * sum(e["success"] for e in last20) / max(len(last20), 1),
                    "mean_return_last20": sum(e["reward"] for e in last20) / max(len(last20), 1),
                    "reasons": dict(Counter(e["reason"] for e in episodes)),
                    "log_age_seconds": max(0, time.time() - (run / "updates.jsonl").stat().st_mtime),
                    "note": "Training episodes only; held-out evaluation is performed at the end.",
                }
                temporary = run / "dashboard.tmp"
                temporary.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
                temporary.replace(run / "dashboard.json")
            writer.flush()
            if status.get("status") in ("completed", "failed"):
                break
            time.sleep(3)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

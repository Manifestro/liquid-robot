"""Describe a frozen-policy transfer test, including measured route distances."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from steppe_robot.benchmark_report import path_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text())
    rows = summary["results"]
    if len(rows) != 50:
        raise ValueError("This transfer gate requires exactly 50 completed worlds")
    successes = sum(row["success"] for row in rows)
    collisions = sum(row["collision"] for row in rows)
    passed = successes >= 48 and collisions <= 1
    distances = [row["start_distance_m"] for row in rows]
    analysis = []
    for row in rows:
        trace = json.loads((run / f"policy_seed_{row['holdout_seed']}_trajectory.json").read_text())
        analysis.append({"seed": row["holdout_seed"], "reason": row["reason"], **path_metrics(trace)})
    (run / "trajectory_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (run / "gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "successes": successes,
                "collisions": collisions,
                "worlds": 50,
                "criterion": "at least 48 successes and at most 1 collision",
                "measured_start_distance_min_m": min(distances),
                "measured_start_distance_max_m": max(distances),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#28876b" if r["success"] else "#c85b43" for r in rows]
    axes[0].scatter(distances, [r["elapsed_seconds"] for r in rows], c=colors)
    axes[0].set(
        xlabel="Measured initial GPS distance (m)",
        ylabel="Attempt duration (s)",
        title="50 new worlds; green = success",
    )
    counts = Counter(row["reason"] for row in rows)
    axes[1].bar(list(counts), list(counts.values()))
    axes[1].set(title="Terminations", ylabel="World count")
    fig.tight_layout()
    fig.savefig(run / "transfer.png", dpi=160)
    plt.close(fig)
    failed = [r for r in rows if not r["success"]]
    selected = sorted(failed or rows, key=lambda r: r["elapsed_seconds"], reverse=True)[:3]
    fig, axes = plt.subplots(1, len(selected), figsize=(12, 4), squeeze=False)
    for axis, row in zip(axes[0], selected):
        trace = json.loads((run / f"policy_seed_{row['holdout_seed']}_trajectory.json").read_text())
        axis.plot([p["x"] for p in trace], [p["z"] for p in trace])
        axis.scatter(trace[0]["x"], trace[0]["z"], color="black", s=20)
        goal = row["goal"]
        axis.scatter(goal["x"], goal["z"], marker="*", s=100, color="#ca9200")
        axis.set(title=f"{row['holdout_seed']}: {row['reason']}", xlabel="X (m)", ylabel="Z (m)")
        axis.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(run / "routes.png", dpi=160)
    plt.close(fig)
    lines = [
        "# GPS 30–50 m transfer test",
        "",
        "Frozen policy; no training during this test.",
        "",
        f"Success: **{successes}/50**. Collisions: **{collisions}/50**. Gate passed: **{passed}**.",
        f"Measured initial GPS distance: {min(distances):.2f}–{max(distances):.2f} m. The generator places goals 30–50 m forward in Z; road curvature adds a lateral component.",
        f"Mean attempt duration (including failures): {summary['mean_episode_seconds']:.2f} s; timeout 120 s.",
        "",
        "Gate registered before evaluation: at least 48/50 successes, at most 1 collision. Worlds are odd seeds 50001–50099, separate from prior checkpoint selection/confirmation. Known north-facing launch, ideal sensors and unchanged physics.",
        "",
        "![Transfer outcomes](transfer.png)",
        "",
        "![Longest unsuccessful routes, or longest successes if none failed](routes.png)",
        "",
        "Per-world observations, actions, initial/final camera images, checkpoint/build/source provenance and trajectory analysis accompany this report. This is a simulator transfer test, not a real-robot deployment validation.",
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"successes": successes, "collisions": collisions, "passed": passed}))


if __name__ == "__main__":
    main()

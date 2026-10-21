"""Render an auditable checkpoint comparison from saved simulator trajectories."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def path_metrics(trace):
    positions = np.array([[p["x"], p["z"]] for p in trace])
    return {
        "path_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "closest_m": min(p["distance"] for p in trace),
        "start_distance_m": trace[0]["distance"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    data = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    selection, confirmation = data["selection"], data["confirmation"]
    winner, final = str(data["winner_step"]), str(data["final_step"])
    selected = next(c for c in manifest["checkpoints"] if c["step"] == int(winner))
    source = Path(selected["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != selected["sha256"]:
        raise ValueError("Selected checkpoint changed after benchmark")
    shutil.copyfile(source, run / "best.pt")
    analysis = {}
    for label, summary in confirmation.items():
        rows = []
        for result in summary["results"]:
            trace = json.loads(
                (run / "confirmation" / label / f"policy_seed_{result['holdout_seed']}_trajectory.json").read_text()
            )
            rows.append({"seed": result["holdout_seed"], "reason": result["reason"], **path_metrics(trace)})
        analysis[label] = rows
    (run / "trajectory_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, summaries, title in [
        (axes[0], selection, "Selection: 12 new worlds"),
        (axes[1], confirmation, "Independent confirmation: 30 worlds"),
    ]:
        labels = list(summaries)
        axis.bar(labels, [100 * summaries[k]["success_rate"] for k in labels], color="#248a68")
        axis.set(title=title, xlabel="Training checkpoint (steps)", ylabel="Success (%)", ylim=(0, 110))
        for i, key in enumerate(labels):
            successes = sum(row["success"] for row in summaries[key]["results"])
            axis.text(
                i, 100 * summaries[key]["success_rate"] + 2, f"{successes}/{summaries[key]['episodes']}", ha="center"
            )
    labels = list(confirmation)
    axes[2].bar(labels, [confirmation[k]["mean_episode_seconds"] for k in labels], color="#3479af")
    axes[2].set(
        title="Confirmation duration (includes failures)",
        xlabel="Training checkpoint (steps)",
        ylabel="Simulation seconds",
    )
    fig.tight_layout()
    fig.savefig(run / "comparison.png", dpi=160)
    plt.close(fig)

    if winner != final:
        candidates = list(zip(confirmation[winner]["results"], confirmation[final]["results"]))
        candidates.sort(
            key=lambda pair: (
                pair[0]["success"] and not pair[1]["success"],
                pair[1]["elapsed_seconds"] - pair[0]["elapsed_seconds"],
            ),
            reverse=True,
        )
        fig, axes = plt.subplots(1, min(3, len(candidates)), figsize=(13, 5), squeeze=False)
        for axis, (best_result, last_result) in zip(axes[0], candidates[:3]):
            seed = best_result["holdout_seed"]
            assert seed == last_result["holdout_seed"]
            for label, color in [(winner, "#16825d"), (final, "#ce583b")]:
                trace = json.loads((run / "confirmation" / label / f"policy_seed_{seed}_trajectory.json").read_text())
                axis.plot([p["x"] for p in trace], [p["z"] for p in trace], label=f"step {label}", color=color)
                axis.scatter(trace[0]["x"], trace[0]["z"], color="black", s=12)
            goal = best_result["goal"]
            axis.scatter(goal["x"], goal["z"], marker="*", s=100, color="#d49a00")
            axis.add_patch(plt.Circle((goal["x"], goal["z"]), 1, fill=False, color="#d49a00"))
            axis.set(title=f"World {seed}", xlabel="X (m)", ylabel="Z (m)")
            axis.set_aspect("equal", adjustable="datalim")
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run / "trajectory_comparison.png", dpi=160)
        plt.close(fig)

    lines = [
        "# Frozen-checkpoint comparison",
        "",
        "No training or weight updates were performed.",
        "",
        "Selection and confirmation use disjoint odd seed sets, the same frozen Unity build, 90 s timeout, known north launch, GPS/gyro navigation and deterministic policies.",
        "",
        "| Phase | Checkpoint | Successes | Collisions | Mean duration, s | Mean return |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for phase, summaries in [("selection", selection), ("confirmation", confirmation)]:
        for label, summary in summaries.items():
            successes = sum(row["success"] for row in summary["results"])
            collisions = sum(row["collision"] for row in summary["results"])
            lines.append(
                f"| {phase} | {label} | {successes}/{summary['episodes']} | {collisions} | {summary['mean_episode_seconds']:.2f} | {summary['mean_return']:.3f} |"
            )
    lines.extend(
        [
            "",
            f"Selected checkpoint: **{winner}**, using only the selection set. `best.pt` is its byte-identical copy.",
            "",
            "The independent confirmation set was not used to choose the checkpoint. These results are limited to this simulator and curriculum; 30 worlds are not a hardware-readiness or universal-generalization guarantee.",
            "",
            "Trajectory analysis (`trajectory_analysis.json`) records path length, closest approach and termination for each confirmation world. Repeated near-goal loops indicate a control failure, but do not alone identify whether heading estimation, policy steering or speed is responsible.",
            "",
            f"Wall time: {data['wall_seconds']:.1f} s.",
            "",
            "![Checkpoint comparison](comparison.png)",
        ]
    )
    if winner != final:
        lines.extend(["", "![Trajectory comparison](trajectory_comparison.png)"])
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

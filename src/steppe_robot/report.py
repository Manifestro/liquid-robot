"""Audit a completed experiment and plot the held-out trajectories."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/experiment")
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text())
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
    initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=True)
    final = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    assert final["step"] == summary["training_steps"]
    assert updates[-1]["step"] == final["step"] and all(a["step"] < b["step"] for a, b in zip(updates, updates[1:]))
    assert all(torch.isfinite(value).all() for value in final["model"].values())
    keys = ["loss", "value_loss", "policy_loss", "entropy", "approx_kl", "grad_norm"]
    assert all(np.isfinite(item[key]) for item in updates for key in keys)
    weight_delta = (
        sum(float((value - initial["model"][key]).square().sum()) for key, value in final["model"].items()) ** 0.5
    )
    assert weight_delta > 0
    baseline, evaluation = summary["baseline"], summary["final_evaluation"]
    fig, axes = plt.subplots(1, len(evaluation["results"]), figsize=(13, 5), squeeze=False)
    rows = []
    for axis, before, after in zip(axes[0], baseline["results"], evaluation["results"]):
        seed = after["holdout_seed"]
        traces = {}
        for phase, color, label in [
            ("baseline", "#888888", "Before training"),
            ("evaluation_final", "#167caa", f"After {final['step']:,} decisions"),
        ]:
            trace = json.loads((run / f"{phase}_seed_{seed}_trajectory.json").read_text())
            traces[phase] = trace
            axis.plot([t["x"] for t in trace], [t["z"] for t in trace], color=color, label=label, linewidth=1.5)
        start = traces["baseline"][0]
        axis.scatter(start["x"], start["z"], marker="o", color="black", label="Start", zorder=4)
        goal = after["goal"]
        axis.scatter(goal["x"], goal["z"], marker="*", s=170, color="#d78a00", label="Goal", zorder=5)
        axis.add_patch(plt.Circle((goal["x"], goal["z"]), 1, fill=False, color="#d78a00"))
        axis.scatter(
            after["end_position"]["x"], after["end_position"]["z"], marker="x", color="#bd3131", label="End", zorder=5
        )
        axis.set(title=f"Held-out seed {seed}: {after['reason']}", xlabel="World X (m)", ylabel="World Z (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        closest = min(t["distance"] for t in traces["evaluation_final"])
        rows.append(
            f"| {seed} | {before['progress_m']:.2f} | {after['progress_m']:.2f} | {closest:.2f} | {after['reason']} |"
        )
    fig.suptitle("Deterministic policy on identical held-out worlds; goal radius = 1 m")
    fig.tight_layout()
    fig.savefig(run / "holdout_trajectories.png", dpi=170)
    plt.close(fig)
    audit = {
        "checkpoint_step": final["step"],
        "ppo_updates": len(updates),
        "optimizer_steps": sum(item["optimizer_steps"] for item in updates),
        "all_metrics_finite": True,
        "all_weights_finite": True,
        "weight_change_l2": weight_delta,
    }
    (run / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    before_success = sum(e.get("success", False) for e in baseline["results"])
    after_success = sum(e.get("success", False) for e in evaluation["results"])
    change = "улучшилась" if evaluation["mean_return"] > baseline["mean_return"] else "ухудшилась"
    best = summary.get("best_evaluation")
    best_note = ""
    if best is not None:
        best_note = (
            f"\nВыбранная по отдельной валидации модель: шаг {summary['best_step']}, файл `best.pt`. "
            f"Независимая итоговая проверка: {best['success_rate']:.1%} успешных попыток, "
            f"{best['collision_rate']:.1%} столкновений. Подробности: `evaluation_best.json`. "
            "Таблица и графики ниже относятся к последнему обновлению (`latest.pt`).\n"
        )
    report = (
        f"""# CfC + PPO: эксперимент {final["step"]} шагов

Завершено ровно **{final["step"]} обучающих действий**, {len(updates)} PPO-обновлений, {audit["optimizer_steps"]} шагов оптимизатора. Устройство указано в manifest.json. Весь запуск с исходной и итоговой оценкой занял {summary["wall_seconds"]:.1f} с. Проверки до/после не входят в бюджет обучающих действий.

{best_note}
За обучение завершено {summary["training_episodes"]} эпизодов: успехов {summary["training_successes"]}, столкновений {summary["training_collisions"]}. Последний незавершённый эпизод не считается успехом или провалом.

На одинаковых отложенных мирах успехов до обучения **{before_success}/{len(baseline["results"])}**, после **{after_success}/{len(evaluation["results"])}**. Средняя награда {change}: {baseline["mean_return"]:.4f} → {evaluation["mean_return"]:.4f}; средний конечный прогресс: {baseline["mean_progress_m"]:.3f} → {evaluation["mean_progress_m"]:.3f} м. Отрицательный прогресс означает, что робот закончил дальше от цели, чем начал.

| Seed | Прогресс до, м | Прогресс после, м | Ближайшее расстояние после, м | Завершение |
|---|---:|---:|---:|---|
"""
        + "\n".join(rows)
        + f"""

Все потери/градиенты и веса конечные, L2 изменения весов = {weight_delta:.4f}; обновление модели произошло. Последовательности, очистка памяти, GAE при таймауте и CPU/CUDA backprop проверены тестами. Это подтверждает работоспособность реализации, но не качество навигации после короткого запуска.

Качество нужно оценивать по успешности и траекториям, а не по снижению потерь. Результаты относятся к задаче и конфигурации из manifest.json. Дополнительное обучение после установленного бюджета {final["step"]} шагов не запускалось.

Геометрия введена по размерам робота; параметры привода, контакта шин и шумов датчиков остаются предварительными. Перенос на реальное устройство этим экспериментом не проверялся.

![Траектории](holdout_trajectories.png)

![Сравнение](holdout_comparison.png)

![Метрики PPO](training_curves.png)

Веса: `latest.pt`; итоговые метрики: `summary.json`; проверка артефактов: `audit.json`. Команды запуска и описание модели: `docs/TRAINING.md` в корне проекта.
"""
    )
    (run / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

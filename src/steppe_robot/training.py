"""Train a recurrent CfC policy using chronological PPO batches and Unity LLAPI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
from importlib.resources import files
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from PIL import Image
from steppe_robot.rl.model import CfCActorCritic
from steppe_robot.rl.ppo import Rollout, compute_gae, update_ppo
from steppe_robot.rl.unity_env import UnityRobotEnv
from steppe_robot.selection import should_replace_best, validate_seed_sets
from steppe_robot.curricula import CURRICULA


@dataclass
class Config:
    env: str = "../steppe-unity/Builds/RobotTraining/RobotTraining.exe"
    run: str = "runs/experiment"
    steps: int = 5000
    rollout_steps: int = 256
    sequence_length: int = 32
    burn_in: int = 8
    sequences_per_batch: int = 4
    epochs: int = 4
    hidden_size: int = 128
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    seed: int = 42
    worker_id: int = 10
    time_scale: float = 20
    timeout: float = 90
    eval_seeds: tuple[int, ...] = tuple(range(30001, 30061, 2))
    validation_seeds: tuple[int, ...] = tuple(range(10001, 10025, 2))
    validation_interval: int = 10000
    device: str = "cuda"
    curriculum: str = "navigation"
    initialize_from: str = ""


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def append_json(path: Path, data):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")


def save_checkpoint(path, model, optimizer, config, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": 1,
            "algorithm": "CfC-sequence-PPO",
            "step": step,
            "config": asdict(config),
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


@torch.no_grad()
def refresh_state(model, history, device):
    """Reconstruct current episode memory after changing the policy weights."""
    state = model.initial_state(1, device)
    for start in range(0, len(history), 64):
        chunk = history[start : start + 64]
        images = torch.as_tensor(np.stack([obs.image for obs in chunk]), device=device)[None]
        vectors = torch.as_tensor(np.stack([obs.vector for obs in chunk]), device=device)[None]
        resets = torch.zeros((1, len(chunk)), dtype=torch.bool, device=device)
        if start == 0:
            resets[0, 0] = True
        _, _, _, state = model.forward_sequence(images, vectors, state, resets)
    return state


def collect(model, env, observation, state, episode_start, history, size, config, run, global_step):
    rollout = Rollout.allocate(size, model.hidden_size)
    episodes = []
    for i in range(size):
        rollout.images[i], rollout.vectors[i] = observation.image, observation.vector
        rollout.states[i] = state[0].cpu().numpy()
        rollout.starts[i] = episode_start
        action, latent, log_prob, value, next_state = model.act(
            observation.image, observation.vector, state, episode_start
        )
        rollout.z[i], rollout.log_probs[i], rollout.values[i] = latent, log_prob, value
        if i > 0 and not rollout.dones[i - 1]:
            rollout.next_values[i - 1] = value
        transition = env.step(action)
        rollout.rewards[i], rollout.dones[i] = transition.reward, transition.done
        if transition.done:
            if transition.interrupted:
                terminal = transition.terminal_observation
                _, _, _, bootstrap, _ = model.act(
                    terminal.image, terminal.vector, next_state, False, deterministic=True
                )
                rollout.next_values[i] = bootstrap
            else:
                rollout.next_values[i] = 0
            state = model.initial_state(1, config.device)
            history = []
            episode_start = True
        else:
            history.append(observation)
            state, episode_start = next_state, False
        observation = transition.observation
        for metric in transition.metrics:
            row = {**metric, "phase": "training", "training_step": global_step + i + 1}
            append_json(run / "episodes.jsonl", row)
            episodes.append(row)
    if not rollout.dones[-1]:
        _, _, _, bootstrap, _ = model.act(
            observation.image, observation.vector, state, episode_start, deterministic=True
        )
        rollout.next_values[-1] = bootstrap
    rollout.advantages, rollout.returns = compute_gae(
        rollout.rewards, rollout.values, rollout.next_values, rollout.dones, config.gamma, config.gae_lambda
    )
    return rollout, observation, state, episode_start, history, episodes


def evaluate(model, env, config, run, phase):
    episodes = []
    for seed in config.eval_seeds:
        write_json(
            run / "evaluation_status.json",
            {"phase": phase, "completed": len(episodes), "total": len(config.eval_seeds), "seed": seed},
        )
        observation = env.reset(evaluation=True, world_seed=seed, timeout=config.timeout)
        state, episode_start = model.initial_state(1, config.device), True
        start_distance = float(observation.vector[1090])
        reward, trajectory = 0.0, []
        Image.fromarray(observation.image.transpose(1, 2, 0)).save(run / f"{phase}_seed_{seed}_start.png")
        for step in range(int(config.timeout * 10) + 20):
            action, _, _, _, state = model.act(
                observation.image, observation.vector, state, episode_start, deterministic=True
            )
            trajectory.append(
                {
                    "x": float(observation.vector[1106]),
                    "z": float(observation.vector[1108]),
                    "distance": float(observation.vector[1090]),
                    "left": float(action[0]),
                    "right": float(action[1]),
                    "goal_right": float(observation.vector[1091]),
                    "goal_forward": float(observation.vector[1092]),
                    "yaw_rate_radps": float(observation.vector[1084]),
                    "gps_course_degrees": float(observation.vector[1112]),
                    "gps_course_valid": bool(observation.vector[1113]),
                    "speed_mps": float(np.linalg.norm(observation.vector[1109:1112])),
                }
            )
            transition = env.step(action)
            reward += transition.reward
            if transition.done:
                final = transition.terminal_observation
                metric = transition.metrics[-1] if transition.metrics else {}
                row = {
                    **metric,
                    "phase": phase,
                    "holdout_seed": seed,
                    "return": reward,
                    "decisions": step + 1,
                    "start_distance_m": start_distance,
                    "end_distance_m": float(final.vector[1090]),
                    "progress_m": start_distance - float(final.vector[1090]),
                }
                Image.fromarray(final.image.transpose(1, 2, 0)).save(run / f"{phase}_seed_{seed}_end.png")
                episodes.append(row)
                write_json(
                    run / "evaluation_status.json",
                    {"phase": phase, "completed": len(episodes), "total": len(config.eval_seeds), "last_result": row},
                )
                append_json(run / "evaluation.jsonl", row)
                write_json(run / f"{phase}_seed_{seed}_trajectory.json", trajectory)
                print(json.dumps({"event": "evaluation_episode", **row}), flush=True)
                break
            observation, episode_start = transition.observation, False
        else:
            raise RuntimeError("Evaluation episode exceeded its Unity timeout")
    result = (
        {
            "episodes": len(episodes),
            "success_rate": float(np.mean([e.get("success", False) for e in episodes])),
            "collision_rate": float(np.mean([e.get("collision", False) for e in episodes])),
            "mean_return": float(np.mean([e["return"] for e in episodes])),
            "mean_progress_m": float(np.mean([e["progress_m"] for e in episodes])),
            "mean_episode_seconds": float(np.mean([e.get("elapsed_seconds", 0) for e in episodes])),
            "results": episodes,
        }
        if episodes
        else {"episodes": 0}
    )
    write_json(run / f"{phase}.json", result)
    return result


def plot_results(run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    steps = [m["step"] for m in metrics]
    for axis, keys, title in [
        (axes[0, 0], ["policy_loss", "value_loss"], "PPO losses"),
        (axes[0, 1], ["entropy"], "Squashed action entropy"),
        (axes[1, 0], ["approx_kl", "clip_fraction"], "Policy update size"),
        (axes[1, 1], ["mean_step_reward"], "Mean transition reward"),
    ]:
        for key in keys:
            axis.plot(steps, [m[key] for m in metrics], marker=".", label=key)
        axis.set(xlabel="Unity action decisions", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(run / "training_curves.png", dpi=160)
    plt.close(fig)
    baseline = json.loads((run / "baseline.json").read_text())
    final = json.loads((run / "evaluation_final.json").read_text())
    if baseline["episodes"] and final["episodes"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = [str(e["holdout_seed"]) for e in baseline["results"]]
        x = np.arange(len(labels))
        for axis, key, title in [
            (axes[0], "progress_m", "Progress toward goal (m)"),
            (axes[1], "return", "Episode return"),
        ]:
            axis.bar(x - 0.18, [e[key] for e in baseline["results"]], 0.36, label="Before training")
            axis.bar(x + 0.18, [e[key] for e in final["results"]], 0.36, label="After training")
            axis.set_xticks(x, labels)
            axis.set(xlabel="Held-out world seed", title=title)
            axis.legend()
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(run / "holdout_comparison.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=Config.env)
    parser.add_argument("--run", default=Config.run)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-id", type=int, default=10)
    parser.add_argument("--time-scale", type=float, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--curriculum", choices=CURRICULA, default="navigation")
    parser.add_argument(
        "--initialize-from", default="", help="Load policy weights, starting a new optimizer and step counter"
    )
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--validation-interval", type=int, default=Config.validation_interval)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=Config.validation_seeds)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=Config.eval_seeds)
    args = parser.parse_args()
    config = Config(
        env=args.env,
        run=args.run,
        steps=args.steps,
        seed=args.seed,
        worker_id=args.worker_id,
        time_scale=args.time_scale,
        device=args.device,
        eval_seeds=() if args.no_eval else tuple(args.eval_seeds),
        validation_seeds=tuple(args.validation_seeds),
        validation_interval=0 if args.no_eval else args.validation_interval,
        learning_rate=args.learning_rate,
        curriculum=args.curriculum,
        timeout=args.timeout,
        initialize_from=args.initialize_from,
    )
    if config.steps < 1:
        parser.error("--steps must be positive")
    if args.validation_interval < 0 or not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        parser.error("Validation interval must be nonnegative and learning rate finite and positive")
    if config.validation_interval:
        try:
            validate_seed_sets(config.validation_seeds, config.eval_seeds)
        except ValueError as error:
            parser.error(str(error))
    run = Path(config.run)
    run.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(4)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly choose --device cpu to use CPU")
    rng = np.random.default_rng(config.seed)
    model = CfCActorCritic(config.hidden_size).to(config.device)
    if config.initialize_from:
        checkpoint = torch.load(config.initialize_from, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    initial_parameters = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    manifest = {
        "config": asdict(config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "ncps": importlib.metadata.version("ncps"),
        "package_version": importlib.metadata.version("steppe-robot-learning"),
        "protocol_sha256": hashlib.sha256(
            files("steppe_robot").joinpath("schemas/observation-v1.json").read_bytes()
        ).hexdigest(),
        "device": torch.cuda.get_device_name() if config.device == "cuda" else "CPU",
        "initial_checkpoint_sha256": hashlib.sha256(Path(config.initialize_from).read_bytes()).hexdigest()
        if config.initialize_from
        else None,
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "unity_assembly_sha256": hashlib.sha256(
            (Path(config.env).parent / "RobotTraining_Data/Managed/Assembly-CSharp.dll").read_bytes()
        ).hexdigest(),
    }
    write_json(run / "manifest.json", manifest)
    save_checkpoint(run / "initial.pt", model, optimizer, config, 0)
    print(
        json.dumps(
            {
                "event": "start",
                "parameters": manifest["parameter_count"],
                "device": manifest["device"],
                "steps": config.steps,
            }
        ),
        flush=True,
    )
    env = validation_env = None
    best_validation, best_step = None, None
    started = time.monotonic()
    try:
        env = UnityRobotEnv(
            config.env,
            worker_id=config.worker_id,
            seed=config.seed,
            time_scale=config.time_scale,
            log_folder=run / "unity-player",
            curriculum=config.curriculum,
            background=True,
        )
        write_json(run / "status.json", {"status": "baseline", "step": 0, "target_steps": config.steps})
        baseline = evaluate(model, env, config, run, "baseline")
        if config.validation_interval:
            validation_env = UnityRobotEnv(
                config.env,
                worker_id=config.worker_id + 100,
                seed=config.seed,
                time_scale=config.time_scale,
                log_folder=run / "validation-player",
                curriculum=config.curriculum,
                background=True,
            )

        def validate(step):
            nonlocal best_validation, best_step
            write_json(run / "status.json", {"status": "validating", "step": step, "target_steps": config.steps})
            directory = run / "validation" / f"step_{step}"
            directory.mkdir(parents=True)
            result = evaluate(
                model, validation_env, replace(config, eval_seeds=config.validation_seeds), directory, "validation"
            )
            improved = should_replace_best(result, best_validation)
            append_json(run / "validation.jsonl", {"step": step, "improved": improved, **result})
            if improved:
                best_validation, best_step = result, step
                save_checkpoint(run / "best.pt", model, optimizer, config, step)
                write_json(run / "best.json", {"step": step, "validation": result, "checkpoint": str(run / "best.pt")})

        if validation_env is not None:
            validate(0)
        observation = env.reset(timeout=config.timeout)
        state = model.initial_state(1, config.device)
        episode_start, history = True, []
        step, update, last_numbered, last_validation = 0, 0, 0, 0
        completed_episodes = []
        training_started = time.monotonic()
        while step < config.steps:
            rollout, observation, state, episode_start, history, episodes = collect(
                model,
                env,
                observation,
                state,
                episode_start,
                history,
                min(config.rollout_steps, config.steps - step),
                config,
                run,
                step,
            )
            completed_episodes.extend(episodes)
            step += len(rollout.rewards)
            metrics = update_ppo(
                model,
                optimizer,
                rollout,
                epochs=config.epochs,
                sequence_length=config.sequence_length,
                burn_in=config.burn_in,
                sequences_per_batch=config.sequences_per_batch,
                clip=config.clip,
                value_coef=config.value_coef,
                entropy_coef=config.entropy_coef,
                max_grad_norm=config.max_grad_norm,
                target_kl=config.target_kl,
                rng=rng,
            )
            state = refresh_state(model, history, config.device)
            update += 1
            metrics.update(
                step=step,
                update=update,
                mean_step_reward=float(rollout.rewards.mean()),
                completed_episodes=len(completed_episodes),
                training_successes=sum(e["success"] for e in completed_episodes),
                mean_rollout_return=float(rollout.returns.mean()),
                decisions_per_second=step / (time.monotonic() - training_started),
                elapsed_seconds=time.monotonic() - started,
            )
            append_json(run / "updates.jsonl", metrics)
            print(json.dumps({"event": "update", **metrics}), flush=True)
            save_checkpoint(run / "latest.pt", model, optimizer, config, step)
            if step - last_numbered >= 1000 or step == config.steps:
                save_checkpoint(run / "checkpoints" / f"step_{step}.pt", model, optimizer, config, step)
                last_numbered = step
            if validation_env is not None and (
                step - last_validation >= config.validation_interval or step == config.steps
            ):
                validate(step)
                last_validation = step
            write_json(run / "status.json", {"status": "training", "step": step, "target_steps": config.steps})
        write_json(run / "status.json", {"status": "evaluating", "step": step, "target_steps": config.steps})
        final = evaluate(model, env, config, run, "evaluation_final")
        best_evaluation = None
        if best_validation is not None:
            best_model = CfCActorCritic(config.hidden_size).to(config.device).eval()
            best_model.load_state_dict(
                torch.load(run / "best.pt", map_location=config.device, weights_only=True)["model"]
            )
            best_evaluation = evaluate(best_model, env, config, run, "evaluation_best")
        delta = (
            sum(float((v.detach().cpu() - initial_parameters[k]).square().sum()) for k, v in model.state_dict().items())
            ** 0.5
        )
        summary = {
            "status": "completed",
            "training_steps": step,
            "ppo_updates": update,
            "training_episodes": len(completed_episodes),
            "training_successes": sum(e["success"] for e in completed_episodes),
            "training_collisions": sum(e["collision"] for e in completed_episodes),
            "baseline": baseline,
            "final_evaluation": final,
            "weight_change_l2": delta,
            "wall_seconds": time.monotonic() - started,
            "checkpoint": str(run / "latest.pt"),
            "best_step": best_step,
            "best_validation": best_validation,
            "best_evaluation": best_evaluation,
            "best_checkpoint": str(run / "best.pt") if best_validation is not None else None,
        }
        write_json(run / "summary.json", summary)
        write_json(run / "status.json", {"status": "completed", "step": step, "target_steps": config.steps})
        plot_results(run)
        print(json.dumps({"event": "completed", **summary}), flush=True)
    except BaseException as exception:
        write_json(run / "failure.json", {"error": repr(exception), "wall_seconds": time.monotonic() - started})
        write_json(run / "status.json", {"status": "failed", "error": repr(exception)})
        raise
    finally:
        if validation_env is not None:
            validation_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

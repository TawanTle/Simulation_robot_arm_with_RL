"""Evaluate PPO, SAC, and TD3 and save a separate dataset for each model.

Run after training:
    python Evaluate_result.py

Examples:
    python Evaluate_result.py --episodes 100
    python Evaluate_result.py --models ppo sac --episodes 50
    python Evaluate_result.py --model-source final
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from Simulation_robot_arm_with_RL.rl_env import PPOArmEnv


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
DEVICE = "auto"

# Hyperparameters used in the three training files.
# PPO, SAC, and TD3 do not use epsilon-greedy, so epsilon decay is N/A.
MODEL_CONFIGS = {
    "ppo": {
        "class": PPO,
        "display_name": "PPO",
        "alpha": 3e-4,
        "gamma": 0.99,
        "epsilon_decay": "N/A",
        "best_path": ROOT / "models" / "ppo" / "best" / "best_model.zip",
        "final_path": ROOT / "robotic_arm_3dof_ppo.zip",
    },
    "sac": {
        "class": SAC,
        "display_name": "SAC",
        "alpha": 3e-4,
        "gamma": 0.99,
        "epsilon_decay": "N/A",
        "best_path": ROOT / "models" / "sac" / "best" / "best_model.zip",
        "final_path": ROOT / "robotic_arm_3dof_sac.zip",
    },
    "td3": {
        "class": TD3,
        "display_name": "TD3",
        "alpha": 3e-4,
        "gamma": 0.99,
        "epsilon_decay": "N/A",
        "best_path": ROOT / "models" / "td3" / "best" / "best_model.zip",
        "final_path": ROOT / "robotic_arm_3dof_td3.zip",
    },
}


STEP_FIELDS = [
    "Model",
    "Episode",
    "Step",
    "Theta1_deg",
    "Theta2_deg",
    "Theta3_deg",
    "Action1",
    "Action2",
    "Action3",
    "End_X_cm",
    "End_Y_cm",
    "End_Z_cm",
    "Target_X_cm",
    "Target_Y_cm",
    "Target_Z_cm",
    "Distance_cm",
    "Reward",
    "Cumulative_Reward",
    "Success",
    "Terminated",
    "Truncated",
    "Inference_Time_ms",
]

EPISODE_FIELDS = [
    "Model",
    "Episode",
    "Seed",
    "Total_Reward",
    "Steps",
    "Success",
    "Initial_Distance_cm",
    "Final_Distance_cm",
    "Episode_Time_s",
    "Mean_Inference_Time_ms",
    "Max_Inference_Time_ms",
]

SUMMARY_FIELDS = [
    "Run_ID",
    "Model",
    "Alpha",
    "Gamma",
    "Epsilon_Decay",
    "Training_Timesteps",
    "Training_Episodes",
    "Training_Time_s",
    "Evaluation_Episodes",
    "Mean_Reward",
    "Std_Reward",
    "Success_Rate_pct",
    "Mean_Episode_Length",
    "Mean_Initial_Distance_cm",
    "Mean_Final_Distance_cm",
    "Total_Evaluation_Time_s",
    "Mean_Episode_Time_s",
    "Mean_Inference_Time_ms",
    "Model_Path",
]


def write_csv(path, fieldnames, rows):
    """Write rows as an Excel-friendly UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_model_path(config, model_source):
    """Choose the best checkpoint first, or the final model when requested."""
    if model_source == "best":
        candidates = [config["best_path"]]
    elif model_source == "final":
        candidates = [config["final_path"]]
    else:
        candidates = [config["best_path"], config["final_path"]]

    for path in candidates:
        if path.exists():
            return path

    checked_paths = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "ไม่พบไฟล์โมเดล กรุณาเทรนโมเดลก่อน โดยตรวจหาใน:\n"
        f"{checked_paths}"
    )


def read_latest_training_record(model_name):
    """Read the latest recorded training time for one model."""
    model_upper = model_name.upper()
    candidates = [
        RESULTS_ROOT / model_name / "training_times.csv",
        RESULTS_ROOT / "training_times.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        with path.open("r", newline="", encoding="utf-8-sig") as file:
            matching_rows = [
                row
                for row in csv.DictReader(file)
                if row.get("Model", "").upper() == model_upper
            ]

        if matching_rows:
            latest = matching_rows[-1]
            return (
                latest.get("Training Timesteps", ""),
                latest.get("Training Time (s)", ""),
            )

    return "", ""


def count_training_episodes(model_name):
    """Count completed episodes from this model's Monitor dataset."""
    model_result_dir = RESULTS_ROOT / model_name
    monitor_files = sorted(model_result_dir.glob("train_monitor*.csv"))

    if not monitor_files:
        return ""

    # Only one training Monitor file is expected. If several exist, use the
    # newest one so an old copied file is not counted twice.
    monitor_path = max(monitor_files, key=lambda path: path.stat().st_mtime)

    with monitor_path.open("r", encoding="utf-8-sig") as file:
        data_lines = [
            line for line in file if line.strip() and not line.lstrip().startswith("#")
        ]

    if len(data_lines) <= 1:
        return 0

    return sum(1 for _ in csv.DictReader(data_lines))


def make_step_row(
    model_name,
    episode,
    step,
    info,
    action,
    reward,
    cumulative_reward,
    terminated,
    truncated,
    inference_time_ms,
):
    """Convert one environment state into a flat CSV row."""
    angles = np.asarray(info["angles"], dtype=float)
    end_position = np.asarray(info["end_position"], dtype=float)
    target_position = np.asarray(info["target_position"], dtype=float)

    if action is None:
        action_values = ["", "", ""]
    else:
        action_values = np.asarray(action, dtype=float).reshape(-1)

    return {
        "Model": model_name,
        "Episode": episode,
        "Step": step,
        "Theta1_deg": round(float(angles[0]), 6),
        "Theta2_deg": round(float(angles[1]), 6),
        "Theta3_deg": round(float(angles[2]), 6),
        "Action1": "" if action is None else round(float(action_values[0]), 6),
        "Action2": "" if action is None else round(float(action_values[1]), 6),
        "Action3": "" if action is None else round(float(action_values[2]), 6),
        "End_X_cm": round(float(end_position[0]), 6),
        "End_Y_cm": round(float(end_position[1]), 6),
        "End_Z_cm": round(float(end_position[2]), 6),
        "Target_X_cm": round(float(target_position[0]), 6),
        "Target_Y_cm": round(float(target_position[1]), 6),
        "Target_Z_cm": round(float(target_position[2]), 6),
        "Distance_cm": round(float(info["distance"]), 6),
        "Reward": round(float(reward), 6),
        "Cumulative_Reward": round(float(cumulative_reward), 6),
        "Success": bool(info.get("success", False)),
        "Terminated": bool(terminated),
        "Truncated": bool(truncated),
        "Inference_Time_ms": (
            "" if inference_time_ms is None else round(float(inference_time_ms), 6)
        ),
    }


def evaluate_model(model_name, episodes, base_seed, model_source, run_id):
    """Evaluate one model and save its step, episode, and summary datasets."""
    config = MODEL_CONFIGS[model_name]
    display_name = config["display_name"]
    model_path = select_model_path(config, model_source)
    model = config["class"].load(str(model_path), device=DEVICE)
    env = PPOArmEnv()

    model_output_dir = RESULTS_ROOT / model_name / "evaluation" / run_id
    step_rows = []
    episode_rows = []
    evaluation_start = time.perf_counter()

    print(f"\nกำลัง Evaluate {display_name}: {model_path}")

    try:
        for episode_index in range(1, episodes + 1):
            episode_seed = base_seed + episode_index - 1
            episode_start = time.perf_counter()
            observation, info = env.reset(seed=episode_seed)

            initial_distance = float(info["distance"])
            cumulative_reward = 0.0
            inference_times = []
            terminated = False
            truncated = False

            # Step 0 stores the initial pose before the first action.
            step_rows.append(
                make_step_row(
                    display_name,
                    episode_index,
                    0,
                    info,
                    action=None,
                    reward=0.0,
                    cumulative_reward=0.0,
                    terminated=False,
                    truncated=False,
                    inference_time_ms=None,
                )
            )

            while not (terminated or truncated):
                prediction_start = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                inference_time_ms = (
                    time.perf_counter() - prediction_start
                ) * 1000.0

                inference_times.append(inference_time_ms)
                observation, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += float(reward)

                step_rows.append(
                    make_step_row(
                        display_name,
                        episode_index,
                        int(info["step_count"]),
                        info,
                        action=action,
                        reward=reward,
                        cumulative_reward=cumulative_reward,
                        terminated=terminated,
                        truncated=truncated,
                        inference_time_ms=inference_time_ms,
                    )
                )

            episode_time = time.perf_counter() - episode_start
            success = bool(info.get("success", False))

            episode_rows.append(
                {
                    "Model": display_name,
                    "Episode": episode_index,
                    "Seed": episode_seed,
                    "Total_Reward": round(cumulative_reward, 6),
                    "Steps": int(info["step_count"]),
                    "Success": success,
                    "Initial_Distance_cm": round(initial_distance, 6),
                    "Final_Distance_cm": round(float(info["distance"]), 6),
                    "Episode_Time_s": round(episode_time, 6),
                    "Mean_Inference_Time_ms": round(
                        float(np.mean(inference_times)), 6
                    ),
                    "Max_Inference_Time_ms": round(
                        float(np.max(inference_times)), 6
                    ),
                }
            )

            print(
                f"  Episode {episode_index:03d}/{episodes}: "
                f"reward={cumulative_reward:9.2f}, "
                f"distance={float(info['distance']):6.3f} cm, "
                f"success={success}"
            )
    finally:
        env.close()

    total_evaluation_time = time.perf_counter() - evaluation_start
    rewards = np.asarray([row["Total_Reward"] for row in episode_rows], dtype=float)
    lengths = np.asarray([row["Steps"] for row in episode_rows], dtype=float)
    successes = np.asarray([row["Success"] for row in episode_rows], dtype=bool)
    initial_distances = np.asarray(
        [row["Initial_Distance_cm"] for row in episode_rows], dtype=float
    )
    final_distances = np.asarray(
        [row["Final_Distance_cm"] for row in episode_rows], dtype=float
    )
    episode_times = np.asarray(
        [row["Episode_Time_s"] for row in episode_rows], dtype=float
    )
    all_inference_times = np.asarray(
        [
            row["Inference_Time_ms"]
            for row in step_rows
            if row["Inference_Time_ms"] != ""
        ],
        dtype=float,
    )

    training_timesteps, training_time = read_latest_training_record(model_name)
    training_episodes = count_training_episodes(model_name)

    summary_row = {
        "Run_ID": run_id,
        "Model": display_name,
        "Alpha": config["alpha"],
        "Gamma": config["gamma"],
        "Epsilon_Decay": config["epsilon_decay"],
        "Training_Timesteps": training_timesteps,
        "Training_Episodes": training_episodes,
        "Training_Time_s": training_time,
        "Evaluation_Episodes": episodes,
        "Mean_Reward": round(float(np.mean(rewards)), 6),
        "Std_Reward": round(float(np.std(rewards)), 6),
        "Success_Rate_pct": round(float(np.mean(successes) * 100.0), 2),
        "Mean_Episode_Length": round(float(np.mean(lengths)), 2),
        "Mean_Initial_Distance_cm": round(float(np.mean(initial_distances)), 6),
        "Mean_Final_Distance_cm": round(float(np.mean(final_distances)), 6),
        "Total_Evaluation_Time_s": round(total_evaluation_time, 6),
        "Mean_Episode_Time_s": round(float(np.mean(episode_times)), 6),
        "Mean_Inference_Time_ms": round(float(np.mean(all_inference_times)), 6),
        "Model_Path": str(model_path),
    }

    step_file = model_output_dir / "evaluation_steps.csv"
    episode_file = model_output_dir / "evaluation_episodes.csv"
    summary_file = model_output_dir / "evaluation_summary.csv"

    write_csv(step_file, STEP_FIELDS, step_rows)
    write_csv(episode_file, EPISODE_FIELDS, episode_rows)
    write_csv(summary_file, SUMMARY_FIELDS, [summary_row])

    print(f"บันทึก Dataset ของ {display_name} ที่: {model_output_dir}")
    return summary_row


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO, SAC, and TD3 on identical robot-arm targets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
        help="Models to evaluate (default: ppo sac td3)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per model (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="First evaluation seed (default: 42)",
    )
    parser.add_argument(
        "--model-source",
        choices=("auto", "best", "final"),
        default="auto",
        help="auto uses best_model.zip first, then the final model",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.episodes <= 0:
        raise ValueError("--episodes ต้องมากกว่า 0")

    # All models share these episode seeds, so they receive identical initial
    # joint angles and targets for a fair comparison.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_rows = []

    for model_name in dict.fromkeys(args.models):
        try:
            summary = evaluate_model(
                model_name=model_name,
                episodes=args.episodes,
                base_seed=args.seed,
                model_source=args.model_source,
                run_id=run_id,
            )
            comparison_rows.append(summary)
        except FileNotFoundError as error:
            print(f"\nข้าม {model_name.upper()}: {error}")

    if not comparison_rows:
        raise SystemExit("ไม่พบโมเดลที่สามารถ Evaluate ได้")

    comparison_file = RESULTS_ROOT / f"evaluation_comparison_{run_id}.csv"
    write_csv(comparison_file, SUMMARY_FIELDS, comparison_rows)

    print("\n" + "=" * 60)
    print("Evaluate เสร็จแล้ว")
    print(f"ตารางเปรียบเทียบรวม: {comparison_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

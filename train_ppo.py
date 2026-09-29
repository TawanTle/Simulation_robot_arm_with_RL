"""Train PPO on the shared 3DOF robot-arm environment."""

import csv
from pathlib import Path
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from Simulation_robot_arm_with_RL.rl_env import PPOArmEnv


SEED = 42
TOTAL_TIMESTEPS = 100_000
DEVICE = "auto"

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "ppo"
RESULT_DIR = ROOT / "results" / "ppo"
FINAL_MODEL = ROOT / "robotic_arm_3dof_ppo"


def record_training_time(seconds):
    summary_file = ROOT / "results" / "training_times.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_file.exists()

    with summary_file.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Training Timesteps", "Training Time (s)"])
        writer.writerow(["PPO", TOTAL_TIMESTEPS, seconds])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "best").mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    check_env(PPOArmEnv(), warn=True)

    train_env = Monitor(
        PPOArmEnv(),
        filename=str(RESULT_DIR / "train_monitor.csv"),
    )
    eval_env = Monitor(
        PPOArmEnv(),
        filename=str(RESULT_DIR / "eval_monitor.csv"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_robot_arm",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(RESULT_DIR),
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [256, 256, 128]},
        verbose=1,
        seed=SEED,
        device=DEVICE,
    )

    print("เริ่มเทรน PPO")
    start_time = time.perf_counter()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
        )
        elapsed_time = time.perf_counter() - start_time

        model.save(str(FINAL_MODEL))
        record_training_time(elapsed_time)

        print(f"เทรน PPO เสร็จใน {elapsed_time:.2f} วินาที")
        print(f"บันทึกโมเดล: {FINAL_MODEL}.zip")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

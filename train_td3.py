"""Train TD3 on the shared 3DOF robot-arm environment."""

import csv
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from rl_env import PPOArmEnv


SEED = 42
TOTAL_TIMESTEPS = 100_000
DEVICE = "auto"

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "td3"
RESULT_DIR = ROOT / "results" / "td3"
FINAL_MODEL = ROOT / "robotic_arm_3dof_td3"


def record_training_time(seconds):
    summary_file = ROOT / "results" / "training_times.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_file.exists()

    with summary_file.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Training Timesteps", "Training Time (s)"])
        writer.writerow(["TD3", TOTAL_TIMESTEPS, seconds])


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

    number_of_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(number_of_actions),
        sigma=0.1 * np.ones(number_of_actions),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="td3_robot_arm",
        save_replay_buffer=True,
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

    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=SEED,
        device=DEVICE,
    )

    print("เริ่มเทรน TD3")
    start_time = time.perf_counter()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
        )
        elapsed_time = time.perf_counter() - start_time

        model.save(str(FINAL_MODEL))
        model.save_replay_buffer(str(ROOT / "td3_replay_buffer.pkl"))
        record_training_time(elapsed_time)

        print(f"เทรน TD3 เสร็จใน {elapsed_time:.2f} วินาที")
        print(f"บันทึกโมเดล: {FINAL_MODEL}.zip")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

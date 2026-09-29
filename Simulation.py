from pathlib import Path
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, TD3

from rl_env import PPOArmEnv
from Robot_sim import plot_robot


# เลือกโมเดล: "ppo", "sac" หรือ "td3"
MODEL_NAME = "sac"

# จำนวน Episode ที่ต้องการดู
EPISODES = 3

# ความเร็วภาพ ยิ่งมากยิ่งช้า
DELAY = 0.05


ROOT = Path(__file__).resolve().parent

MODEL_CONFIG = {
    "ppo": {
        "class": PPO,
        "best": ROOT / "models" / "ppo" / "best" / "best_model.zip",
        "final": ROOT / "robotic_arm_3dof_ppo.zip",
    },
    "sac": {
        "class": SAC,
        "best": ROOT / "models" / "sac" / "best" / "best_model.zip",
        "final": ROOT / "robotic_arm_3dof_sac.zip",
    },
    "td3": {
        "class": TD3,
        "best": ROOT / "models" / "td3" / "best" / "best_model.zip",
        "final": ROOT / "robotic_arm_3dof_td3.zip",
    },
}


config = MODEL_CONFIG[MODEL_NAME]

# เลือก best_model ก่อน ถ้าไม่มีจึงใช้ final model
if config["best"].exists():
    model_path = config["best"]
elif config["final"].exists():
    model_path = config["final"]
else:
    raise FileNotFoundError(
        f"ไม่พบโมเดล {MODEL_NAME.upper()} กรุณาเทรนก่อน"
    )

model = config["class"].load(str(model_path))
env = PPOArmEnv()

plt.ion()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for episode in range(1, EPISODES + 1):
    observation, info = env.reset(seed=42 + episode - 1)

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)

        # บันทึกมุมกล้องเดิม
        elevation = ax.elev
        azimuth = ax.azim

        # วาดแขนกล
        plot_robot(
            info["angles"][0],
            info["angles"][1],
            info["angles"][2],
            ax=ax
        )

        # วาดตำแหน่งเป้าหมาย
        target = info["target_position"]

        ax.scatter(
            target[0],
            target[1],
            target[2],
            color="green",
            marker="*",
            s=250,
            label="Target"
        )

        ax.view_init(
            elev=elevation,
            azim=azimuth
        )

        ax.set_title(
            f"{MODEL_NAME.upper()} - Episode {episode}\n"
            f"Step: {info['step_count']} | "
            f"Distance: {info['distance']:.2f} cm | "
            f"Reward: {total_reward:.2f}"
        )

        ax.legend()

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(DELAY)

    print(
        f"Episode {episode}: "
        f"Reward={total_reward:.2f}, "
        f"Distance={info['distance']:.2f}, "
        f"Success={info['success']}"
    )

env.close()

plt.ioff()
plt.show()
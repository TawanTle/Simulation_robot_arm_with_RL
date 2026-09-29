"""Shared Gymnasium environment for the 3DOF robot arm.

This file does not modify Robot_sim.py. It imports and reuses the original
forward_kinematics() function and link lengths from that file.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from Simulation_robot_arm_with_RL.Robot_sim import forward_kinematics, L1, L2, L3


JOINT_LOW = np.array([-90.0, -10.0, -120.0], dtype=np.float32)
JOINT_HIGH = np.array([90.0, 90.0, 0.0], dtype=np.float32)


class PPOArmEnv(gym.Env):
    """Continuous-control environment shared by PPO, SAC, and TD3."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Action = angle change command for joints 1, 2, and 3.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Observation = normalized angles (3), end position (3),
        # target position (3), and target error (3).
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        self.max_steps = 150
        self.max_angle_change = 4.0
        self.touch_threshold = 0.7

        self.current_angles = None
        self.target_position = None
        self.previous_distance = None
        self.step_count = 0

    def end_position(self):
        return forward_kinematics(*self.current_angles)[-1].astype(np.float32)

    def random_target(self):
        # Sampling valid joint angles guarantees a reachable target.
        while True:
            target_angles = self.np_random.uniform(
                JOINT_LOW,
                JOINT_HIGH,
            ).astype(np.float32)

            target = forward_kinematics(*target_angles)[-1].astype(np.float32)
            horizontal = np.hypot(target[0], target[1])

            if target[2] >= 2.0 and horizontal >= 5.0:
                return target

    def observation(self):
        end = self.end_position()
        center = (JOINT_HIGH + JOINT_LOW) / 2.0
        half_range = (JOINT_HIGH - JOINT_LOW) / 2.0
        scale = L1 + L2 + L3

        return np.concatenate(
            [
                (self.current_angles - center) / half_range,
                end / scale,
                self.target_position / scale,
                (self.target_position - end) / scale,
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.current_angles = self.np_random.uniform(
            JOINT_LOW,
            JOINT_HIGH,
        ).astype(np.float32)

        self.target_position = self.random_target()
        self.previous_distance = float(
            np.linalg.norm(self.target_position - self.end_position())
        )

        info = {
            "distance": self.previous_distance,
            "success": False,
            "angles": self.current_angles.copy(),
            "end_position": self.end_position().copy(),
            "target_position": self.target_position.copy(),
            "step_count": self.step_count,
        }

        return self.observation(), info

    def step(self, action):
        self.step_count += 1
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        self.current_angles = np.clip(
            self.current_angles + action * self.max_angle_change,
            JOINT_LOW,
            JOINT_HIGH,
        ).astype(np.float32)

        end = self.end_position()
        distance = float(np.linalg.norm(self.target_position - end))
        improvement = self.previous_distance - distance

        reward = 15.0 * improvement
        reward -= 0.03 * distance
        reward -= 0.01 * float(np.sum(action**2))

        success = distance <= self.touch_threshold
        if success:
            reward += 100.0

        terminated = success
        truncated = self.step_count >= self.max_steps
        self.previous_distance = distance

        info = {
            "distance": distance,
            "success": success,
            "angles": self.current_angles.copy(),
            "end_position": end.copy(),
            "target_position": self.target_position.copy(),
            "step_count": self.step_count,
        }

        return self.observation(), reward, terminated, truncated, info


# Clearer alias for new code while retaining the name used in the notebook.
RobotArmEnv = PPOArmEnv
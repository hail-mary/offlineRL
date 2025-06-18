import gymnasium as gym
import numpy as np
import cv2

class ImageObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=False):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(
            height, width, 1 if grayscale else 3), dtype=np.uint8)

    # def observation(self, obs):
    #     img = self.render(mode='rgb_array')
    #     img = cv2.resize(img, (self.width, self.height))
    #     if self.grayscale:
    #         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #         img = np.expand_dims(img, axis=-1)
    #     return img

import os
import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation
import multiprocessing
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.callbacks import CheckpointCallback

from torch import nn
import torch

# カスタムCNNポリシー（画像観測に対応）
# class CustomCNNPolicy(SAC.policy_aliases):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs,
#                          features_extractor_class=NatureCNN,
#                          features_extractor_kwargs=dict(features_dim=256))

# 学習関数（1つの環境）
def train_env(env_id, logdir="logs", steps=100_000):
    print(f"Training on {env_id}")
    env = gym.make(env_id, render_mode='rgb_array')
    env = ImageObservationWrapper(env)
    env = AddRenderObservation(env, render_only=True)

    model = SAC("CnnPolicy", env, verbose=1, tensorboard_log=f"{logdir}/{env_id}")
    model.learn(total_timesteps=steps)
    model.save(f"{logdir}/{env_id}_sac_model")
    env.close()

if __name__ == '__main__':
    env_ids = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5", "Walker2d-v5"]
    processes = []
    for env_id in env_ids:
        p = multiprocessing.Process(target=train_env, args=(env_id,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

import gymnasium as gym
import numpy as np

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import RoutingEnv
import os

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

# Parallel environments
env = RoutingEnv()
check_env(env, warn=True)

# print(f"training with {device}")
# model = PPO("MultiInputPolicy", env, verbose=1, device="mps")
model = PPO("MultiInputPolicy", env, verbose=1, device="cpu")
if os.path.isfile(f'./models/2layer_mazesolver_multiagent.zip'):
    print(f"... loading multiagent 2layer mazesolver ...")
    model = PPO.load(f"./models/2layer_mazesolver_multiagent", env=env)

for i in range(5):
    model.learn(total_timesteps=20000)
    print("... checkpoint saved ...")
    model.save(f"./models/checkpoint_{i}")
    model.save("./models/2layer_mazesolver_multiagent")

del model
model = PPO.load("./models/2layer_mazesolver_multiagent")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # env.render()

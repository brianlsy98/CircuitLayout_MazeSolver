import gymnasium as gym
import numpy as np

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import RoutingEnv
import os

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

# Parallel environments
env = RoutingEnv(
    start_point        =   np.array([2, 3, 2], dtype=np.float32),
    end_point          =   np.array([6, 5, 3], dtype=np.float32)
)
check_env(env, warn=True)

# print(f"training with {device}")
# model = PPO("MlpPolicy", env, verbose=1, device="mps")
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
if os.path.isfile(f'./models/nlayer_mazesolver.zip'):
    print(f"load nlayer mazesolver.zip")
    model = PPO.load(f"./models/nlayer_mazesolver", env=env)

model.learn(total_timesteps=10000*(abs(env.start_point-env.goal_point)[0]+abs(env.start_point-env.goal_point)[1]))
model.save("./models/nlayer_mazesolver")

del model
model = PPO.load("./models/nlayer_mazesolver")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # env.render()

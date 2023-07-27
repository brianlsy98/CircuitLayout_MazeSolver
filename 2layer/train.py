import gymnasium as gym
import numpy as np

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import RoutingEnv

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

# Parallel environments
env = RoutingEnv()

from stable_baselines3.common.env_checker import check_env
check_env(env, warn=True)


from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000*(abs(env.start_point-env.goal_point)[0]+abs(env.start_point-env.goal_point)[1]))
model.save("./models/2layer_mazesolver")

del model
model = PPO.load("./models/2layer_mazesolver")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()



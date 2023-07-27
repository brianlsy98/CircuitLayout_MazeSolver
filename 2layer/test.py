import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import RoutingEnv

# Parallel environments
env = RoutingEnv()
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
model = PPO.load("./models/2layer_mazesolver")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    traj = info["trajectory"]
    print()
    print("=====================================")
    env.render()
    print("=====================================")

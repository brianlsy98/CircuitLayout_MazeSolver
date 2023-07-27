import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import RoutingEnv

# Parallel environments
env = RoutingEnv(
    start_point        =   np.array([2, 3, 2], dtype=np.float32),
    end_point          =   np.array([6, 5, 3], dtype=np.float32)
)
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
model = PPO.load("./models/nlayer_mazesolver")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    traj = info["trajectory"]

    print(f"obs         :\n {obs}")
    print(f"trajectory  :\n {traj}")
    print(f"done        : {done}")
    env.render()

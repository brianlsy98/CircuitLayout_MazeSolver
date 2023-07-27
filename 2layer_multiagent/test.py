import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import RoutingEnv

# Parallel environments
env = RoutingEnv()
check_env(env, warn=True)

model = PPO("MultiInputPolicy", env, verbose=1)
model = PPO.load("./models/2layer_mazesolver_multiagent")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    traj = info["trajectories"]
    vts = info["via_tags"]

    print()
    print("obs         :")
    for k, v in obs.items():
        print(k)
        print(v)
    print()
    print("trajectories:")
    for t in traj:
        print(t)
    print()
    print(f"via_tags    :")
    for vt in vts:
        print(vt)
    print(f"done        : {done}")
    env.render()

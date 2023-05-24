import gymnasium as gym
import numpy as np

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import RoutingEnv

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

# Parallel environments
env = RoutingEnv(
    init_metal_nodes   =  { "M1": np.array([[2, 3], [2, 5], [6, 3], [6, 5],
                                            [1, 1], [1, 2], [3, 1], [3, 2], [5, 1], [5, 2], [7, 1], [7, 2]]),\
                            "M2": np.array([[1, 1], [2, 1], [3, 1], [1, 3], [2, 3], [3, 3], [5, 3], [6, 3], [7, 3],\
                                            [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0],\
                                            [0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8], [8, 8], [9, 8], [10, 8], [11, 8]]),\
                            "M3": np.array([[]]),\
                            "M4": np.array([[]])},
    init_metal_edges   =  { "M1": np.array([[1, 1.5], [3, 1.5], [5, 1.5], [7, 1.5]]),\
                            "M2": np.array([[1.5, 1], [2.5, 1], [1.5, 3], [2.5, 3], [5.5, 3], [6.5, 3],\
                                            [0.5, 0], [1.5, 0], [2.5, 0], [3.5, 0], [4.5, 0], [5.5, 0], [6.5, 0], [7.5, 0], [8.5, 0], [9.5, 0], [10.5, 0],\
                                            [0.5, 8], [1.5, 8], [2.5, 8], [3.5, 8], [4.5, 8], [5.5, 8], [6.5, 8], [7.5, 8], [8.5, 8], [9.5, 8], [10.5, 8]]),\
                            "M3": np.array([[]]),\
                            "M4": np.array([[]])},
    start_point        =   np.array([2, 3]),
    start_layer        =   1,
    end_point          =   np.array([6, 5]),
    end_layer          =   1,
    render_mode        =   "console"
)
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
# for i in range(env.min_x+2, env.max_x-1, 2):
#     for j in range(env.min_y+2, env.max_y-1, 2):
#         env.start_point = np.array([i, j]); env.end_point = np.array([env.max_x - i, env.max_y - j])
#         if [env.start_point[0], env.start_point[1]] == [env.end_point[0], env.end_point[1]]:
#             env.end_point += np.array([1, 0])
#         model.set_env(env)
#         model.learn(total_timesteps=100000)


model.learn(total_timesteps=15000*(abs(env.start_point-env.goal_point)[0]+abs(env.start_point-env.goal_point)[1]))
model.save("./models/nlayer_mazesolver")

del model
model = PPO.load("./models/nlayer_mazesolver")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # env.render()

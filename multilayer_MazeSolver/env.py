import numpy as np
import gym
from gym import spaces
import copy

class RoutingEnv(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(self,
                 init_metal_nodes   =  {"M1": np.array([[2, 3], [2, 5], [6, 3], [6, 5],
                                                        [1, 1], [1, 2], [3, 1], [3, 2], [5, 1], [5, 2], [7, 1], [7, 2]]),\
                                        "M2": np.array([[1, 1], [2, 1], [3, 1], [1, 3], [2, 3], [3, 3], [5, 3], [6, 3], [7, 3],\
                                                        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0],\
                                                        [0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8], [8, 8], [9, 8], [10, 8], [11, 8]])},
                 init_metal_edges   =  {"M1": np.array([[1, 1.5], [3, 1.5], [5, 1.5], [7, 1.5]]),\
                                        "M2": np.array([[1.5, 1], [2.5, 1], [1.5, 3], [2.5, 3], [5.5, 3], [6.5, 3],\
                                                        [0.5, 0], [1.5, 0], [2.5, 0], [3.5, 0], [4.5, 0], [5.5, 0], [6.5, 0], [7.5, 0], [8.5, 0], [9.5, 0], [10.5, 0],\
                                                        [0.5, 8], [1.5, 8], [2.5, 8], [3.5, 8], [4.5, 8], [5.5, 8], [6.5, 8], [7.5, 8], [8.5, 8], [9.5, 8], [10.5, 8]])},
                 start_point        =   np.array([2, 5]),
                 end_point          =   np.array([6, 3]),
                 routing_grid       =   "12",
                 render_mode        =   "console"):


        super(RoutingEnv, self).__init__()
        self.render_mode            =   render_mode
        self.init_metal_nodes       =   init_metal_nodes
        self.init_metal_edges       =   init_metal_edges
        self.metal_nodes            =   copy.deepcopy(self.init_metal_nodes)
        self.metal_edges            =   copy.deepcopy(self.init_metal_edges)
        self.start_point            =   start_point
        self.goal_point             =   end_point
        self.routing_grid           =   routing_grid
        self.moving_point           =   copy.deepcopy(self.start_point)
        self.moving_metal           =   int(routing_grid[0])
        self.trajectory             =   np.array([[self.moving_metal, self.start_point[0], self.start_point[1]]])
        self.prv_point              =   copy.deepcopy(self.moving_point)
        self.reward_mode            =   "coarse"
        self.goal_reached           =   False
        self.obstacle_reached       =   False


        self.max_x, self.max_y = start_point[0], start_point[1]
        self.min_x, self.min_y = start_point[0], start_point[1]
        for array in self.metal_nodes.values():
            try:
                for i in range(len(array)):
                    if array[i][0] >= self.max_x: self.max_x = array[i][0]
                    if array[i][1] >= self.max_y: self.max_y = array[i][1]
                    if array[i][0] <= self.min_x: self.min_x = array[i][0]
                    if array[i][1] <= self.min_y: self.min_y = array[i][1]
            except:
                pass
        
        self.grid_array             =   np.zeros((self.max_x - self.min_x + 1, self.max_y - self.min_y + 1))
        self.grid_shape             =   np.shape(self.grid_array)
        self.grid_arrays            =   dict()
        for key, value in self.metal_nodes.items():
            self.grid_arrays[key]   =   {"node": copy.deepcopy(self.grid_array),\
                                         "edge": self.metal_edges[key]}
            for i in range(len(value)):
                try: self.grid_arrays[key]["node"][value[i][0]][value[i][1]] = 1
                except: pass

        # Define action and observation space
        # They must be gym.spaces objects
        n_metal_transition = 3 # upper, lower, stay
        n_move = 2 # +1 or -1
        self.action_space = spaces.Discrete(n_metal_transition * n_move)
        self.observation_space = spaces.Box(
            low=-max(self.max_x, self.max_y), high=max(self.max_x, self.max_y), shape=(11,), dtype=np.float32
        )


    def get_obs(self):
        # Vector to goal from moving point
        vec_to_goal = self.goal_point - self.moving_point
        start_to_goal = self.goal_point - self.start_point
        # Vectors to nearest obstacles from metal 1, 2, 3, ...
        # M1, M3, ...
        if self.moving_metal % 2 == 1:
            # current layer nearest obstacle
            vec_moving_layer_obstacle = np.array([0, vec_to_goal[1]])
            for y in range(vec_to_goal[1]):
                if self.grid_arrays[f"M{self.moving_metal}"]["node"]\
                    [self.moving_point[0]][self.moving_point[1] + y] == 1:
                        vec_moving_layer_obstacle = np.array([0, y]); break
            # lower layer nearest obstacle
            vec_prev_layer_obstacle = np.array([vec_to_goal[0], 0])
            for x_ in range(vec_to_goal[0]):
                if f"M{self.moving_metal-1}" in self.grid_arrays.keys():
                    if self.grid_arrays[f"M{self.moving_metal-1}"]["node"]\
                        [self.moving_point[0] + x_][self.moving_point[1]] == 1:
                            vec_prev_layer_obstacle = np.array([x_, 0]); break
            # upper layer nearest obstacle
            vec_next_layer_obstacle = np.array([vec_to_goal[0], 0])
            for x_ in range(vec_to_goal[0]):
                if f"M{self.moving_metal+1}" in self.grid_arrays.keys():
                    if self.grid_arrays[f"M{self.moving_metal+1}"]["node"]\
                        [self.moving_point[0] + x_][self.moving_point[1]] == 1:
                            vec_next_layer_obstacle = np.array([x_, 0]); break
        # M2, M4, ...
        else:
            # current layer nearest obstacle
            vec_moving_layer_obstacle = np.array([vec_to_goal[0], 0])
            for x in range(vec_to_goal[0]):
                if self.grid_arrays[f"M{self.moving_metal}"]["node"]\
                    [self.moving_point[0] + x][self.moving_point[1]] == 1:
                        vec_moving_layer_obstacle = np.array([x, 0]); break
            # lower layer nearest obstacle
            vec_prev_layer_obstacle = np.array([0, vec_to_goal[1]])
            for y_ in range(vec_to_goal[1]):
                if f"M{self.moving_metal-1}" in self.grid_arrays.keys():
                    if self.grid_arrays[f"M{self.moving_metal-1}"]["node"]\
                        [self.moving_point[0]][self.moving_point[1] + y_] == 1:
                            vec_prev_layer_obstacle = np.array([0, y_]); break
            # upper layer nearest obstacle
            vec_next_layer_obstacle = np.array([0, vec_to_goal[1]])
            for y_ in range(vec_to_goal[1]):
                if f"M{self.moving_metal+1}" in self.grid_arrays.keys():
                    if self.grid_arrays[f"M{self.moving_metal+1}"]["node"]\
                        [self.moving_point[0]][self.moving_point[1] + y_] == 1:
                            vec_next_layer_obstacle = np.array([0, y_]); break

        # Return Observation (normalize)
        obs = np.array([(self.moving_point[0]-self.min_x)/(self.max_x-self.min_x), (self.moving_point[1]-self.min_y)/(self.max_y-self.min_y),\
                        self.moving_metal/len(self.grid_arrays),\
                        vec_to_goal[0]/(abs(start_to_goal[0])+1), vec_to_goal[1]/(abs(start_to_goal[1])+1),\
                        vec_moving_layer_obstacle[0]/(abs(start_to_goal[0])+1), vec_moving_layer_obstacle[1]/(abs(start_to_goal[1])+1),\
                        vec_prev_layer_obstacle[0]/(abs(start_to_goal[0])+1), vec_prev_layer_obstacle[1]/(abs(start_to_goal[1])+1),\
                        vec_next_layer_obstacle[0]/(abs(start_to_goal[0])+1), vec_next_layer_obstacle[1]/(abs(start_to_goal[1])+1)], dtype=np.float32)
        
        return obs
    

    def check_termination(self):
        self.goal_reached = bool(self.moving_point[0] == self.goal_point[0]\
                        and self.moving_point[1] == self.goal_point[1])
        self.obstacle_reached = bool(self.grid_arrays[f"M{self.moving_metal}"]["node"]\
                                [self.moving_point[0]][self.moving_point[1]] == 1\
                                or [self.prv_point[0], self.prv_point[1]] == [self.moving_point[0], self.moving_point[1]])
        
        for edge in self.grid_arrays[f"M{self.moving_metal}"]["edge"]:
            if len(edge) != 0:
                if [edge[0], edge[1]] == [(self.moving_point[0]+self.prv_point[0])/2,\
                                        (self.moving_point[1]+self.prv_point[1])/2]:
                    self.obstacle_reached = False; break

        if [self.moving_point[0], self.moving_point[1]] == [self.start_point[0], self.start_point[1]]:
            self.obstacle_reached = False
        
        # when the agent goes to the point in the trajectory
        if [self.moving_point[0], self.moving_point[1]] in [list(traj[1:]) for traj in self.trajectory[:-1]]:
            self.obstacle_reached = True


    def get_rwd(self, action):

        start_goal_dist = np.linalg.norm(self.goal_point - self.start_point)
        agent_goal_dist = np.linalg.norm(self.goal_point - self.moving_point)
        prv_goal_dist = np.linalg.norm(self.goal_point - self.prv_point)

        reward = - (agent_goal_dist - prv_goal_dist)

        if action in [2, 3]:
            reward += 0
        else:
            reward += -1

        if self.obstacle_reached:
            reward -= start_goal_dist**2

        return reward


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent
        self.moving_point = copy.deepcopy(self.start_point)
        self.moving_metal = int(self.routing_grid[0])
        self.trajectory   = np.array([[self.moving_metal, self.start_point[0], self.start_point[1]]])
        self.prv_point    = copy.deepcopy(self.moving_point)
        self.reward_mode  = "coarse"

        obs = self.get_obs()

        return obs  # empty info dict



    def step(self, action):
        # action == 0:   # lower layer, -1      action == 1:   # lower layer, +1
        # action == 2:   # stay, -1             action == 3:   # stay, +1
        # action == 4:   # upper, -1            action == 5:   # upper, +1

        # moving_metal 
        self.moving_metal += action//2 - 1
        if self.moving_metal < 1: self.moving_metal = 1
        elif self.moving_metal > len(self.grid_arrays): self.moving_metal = len(self.grid_arrays) - 1
        
        # moving point
        self.moving_point += np.array([0 if self.moving_metal % 2 == 1 else 2*(action % 2) - 1,
                                       0 if self.moving_metal % 2 == 0 else 2*(action % 2) - 1])

        # Account for the boundaries of the grid
        if self.moving_point[0] > self.max_x: self.moving_point[0] = self.max_x
        if self.moving_point[1] > self.max_y: self.moving_point[1] = self.max_y
        if self.moving_point[0] < self.min_x: self.moving_point[0] = self.min_x
        if self.moving_point[1] < self.min_y: self.moving_point[1] = self.min_y

        # trajectory
        self.trajectory = np.append(self.trajectory, np.array([self.moving_metal,\
                                                               self.moving_point[0],\
                                                               self.moving_point[1]]).reshape(1, 3), axis=0)

        # observation
        obs = self.get_obs()

        # termination condition
        self.check_termination()

        # REWARD
        reward = self.get_rwd(action)

        # done
        done = bool(self.goal_reached or self.obstacle_reached)

        # Optionally we can pass additional info, we are not using that for now
        info = {"trajectory": self.trajectory}

        self.prv_point = copy.deepcopy(self.moving_point)

        return obs, reward, done, info

        



    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            for key, value in self.grid_arrays.items():
                print(f"{key} table")
                print(f"moving metal : {self.moving_metal}")
                for j in range(self.grid_shape[1]-1, -1, -1):
                    for i in range(self.grid_shape[0]):
                        if [i, j] == list(self.moving_point):
                            print("@", end="")
                            for p in value["edge"]:
                                if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                    print("-", end=""); break
                                elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                    print(" ", end="")
                        elif [i, j] == list(self.start_point):
                            print("S", end="")
                            for p in value["edge"]:
                                if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                    print("-", end=""); break
                                elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                    print(" ", end="")
                        elif [i, j] == list(self.goal_point):
                            print("E", end="")
                            for p in value["edge"]:
                                if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                    print("-", end=""); break
                                elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                    print(" ", end="")
                        elif [i, j] in [list(traj[1:]) for traj in self.trajectory]:
                            print(self.trajectory[[list(traj[1:]) for traj in self.trajectory].index([i, j])][0], end="")
                            for p in value["edge"]:
                                if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                    print("-", end=""); break
                                elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                    print(" ", end="")
                        elif value["node"][i][j] == 1:
                            print("X", end="")
                            for p in value["edge"]:
                                if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                    print("-", end=""); break
                                elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                    print(" ", end="")
                        else: print("O", end=" ")
                    print()
                    for i in range(self.grid_shape[0]):
                        for p in value["edge"]:
                            if list(p) != []:
                                if p[0] == i and p[1] == j-0.5:
                                    print("|", end=""); break
                                elif p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]:
                                    print(" ", end="")
                        print("", end=" ")
                    print()
                print()


    def close(self):
        pass


from stable_baselines3.common.env_checker import check_env
if __name__ == "__main__":
    env = RoutingEnv()
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)

    obs = env.reset()
    done = False
    while not done:
        action = np.random.randint(6)
        obs, rewards, done, info = env.step(action)
        env.render()

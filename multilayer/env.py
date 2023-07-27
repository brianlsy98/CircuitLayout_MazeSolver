import numpy as np
import gym
from gym import spaces
import copy

# successful

class RoutingEnv(gym.Env):
    def __init__(self,
                 init_metal_nodes   =   np.array([[2, 3, 1], [2, 5, 1], [6, 3, 1], [6, 5, 1],\
                                                  [1, 1, 1], [1, 2, 1], [3, 1, 1], [3, 2, 1], [5, 1, 1], [5, 2, 1], [7, 1, 1], [7, 2, 1],\
                                                  [1, 1, 2], [2, 1, 2], [3, 1, 2], [1, 3, 2], [2, 3, 2], [3, 3, 2], [5, 3, 2], [6, 3, 2], [7, 3, 2],\
                                                  [0, 0, 2], [1, 0, 2], [2, 0, 2], [3, 0, 2], [4, 0, 2], [5, 0, 2], [6, 0, 2], [7, 0, 2], [8, 0, 2], [9, 0, 2], [10, 0, 2], [11, 0, 2],\
                                                  [0, 8, 2], [1, 8, 2], [2, 8, 2], [3, 8, 2], [4, 8, 2], [5, 8, 2], [6, 8, 2], [7, 8, 2], [8, 8, 2], [9, 8, 2], [10, 8, 2], [11, 8, 2],\
                                                  [5, 3, 4], [6, 3, 4], [7, 3, 4]], dtype=np.float32),
                 init_metal_edges   =   np.array([[1, 1.5, 1], [3, 1.5, 1], [5, 1.5, 1], [7, 1.5, 1],
                                                  [1.5, 1, 2], [2.5, 1, 2], [1.5, 3, 2], [2.5, 3, 2], [5.5, 3, 2], [6.5, 3, 2],
                                                  [0.5, 0, 2], [1.5, 0, 2], [2.5, 0, 2], [3.5, 0, 2], [4.5, 0, 2], [5.5, 0, 2], [6.5, 0, 2], [7.5, 0, 2], [8.5, 0, 2], [9.5, 0, 2], [10.5, 0, 2],\
                                                  [0.5, 8, 2], [1.5, 8, 2], [2.5, 8, 2], [3.5, 8, 2], [4.5, 8, 2], [5.5, 8, 2], [6.5, 8, 2], [7.5, 8, 2], [8.5, 8, 2], [9.5, 8, 2], [10.5, 8, 2],\
                                                  [5.5, 3, 4], [6.5, 3, 4]], dtype=np.float32),
                 start_point        =   np.array([2, 5, 2], dtype=np.float32),
                 end_point          =   np.array([6, 3, 3], dtype=np.float32),
                 obs_search_dist    =   2,
                 render_mode        =   "console"):


        super(RoutingEnv, self).__init__()
        self.metal_nodes            =   copy.deepcopy(init_metal_nodes)
        self.metal_edges            =   copy.deepcopy(init_metal_edges)
        self.start_point            =   copy.deepcopy(start_point)
        self.start_points           =   self.get_routed_points(self.start_point, self.metal_edges)
        self.goal_point             =   copy.deepcopy(end_point)
        self.goal_points            =   self.get_routed_points(self.goal_point, self.metal_edges)
        self.start_point, self.goal_point\
                                    =   self.update_start_goal_points(self.start_points, self.goal_points)
        
        self.obs_search_dist        =   obs_search_dist

        self.render_mode            =   render_mode

        self.moving_point           =   copy.deepcopy(self.start_point)
        self.prv_point              =   copy.deepcopy(self.moving_point)
        self.prv_prv_point          =   copy.deepcopy(self.prv_point)
        self.cur_layer_bool         =   1 if self.moving_point[-1] % 2 == 0 else -1

        self.trajectory             =   np.array([self.start_point], dtype=np.float32)
        self.via_tag                =   [False]

        self.goal_reached           =   False
        self.obstacle_reached       =   False

        # update binary grid arrays
        self.max_x, self.max_y = start_point[0], start_point[1]
        self.min_x, self.min_y = start_point[0], start_point[1]
        for point in self.metal_nodes:
            if point[0] >= self.max_x: self.max_x = point[0]
            if point[1] >= self.max_y: self.max_y = point[1]
            if point[0] <= self.min_x: self.min_x = point[0]
            if point[1] <= self.min_y: self.min_y = point[1]
        self.update_grid_arrays()

        # observation & action space
        # 0 : go front, 1 : go back, 2 : go up, 3 : go down
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
                # cur_layer_bool, vec_to_goal, near_obstacles
            shape=(int(1    +   2\
                        +   6*self.obs_search_dist**2-2*self.obs_search_dist+3),), dtype=np.float32
        )


    def update_grid_arrays(self):
        self.grid_array             =   np.zeros((int(self.max_x - self.min_x + 1), int(self.max_y - self.min_y + 1)), dtype=np.float32)
        self.grid_shape             =   np.shape(self.grid_array)
        self.grid_arrays            =   dict()
        for key in range(int(min([p[-1] for p in self.metal_nodes])), int(max([p[-1] for p in self.metal_nodes]) + 1)):
            self.grid_arrays[f"M{key}"] = {"node": copy.deepcopy(self.grid_array),\
                                           "edge": np.array([edge[:-1] for edge in self.metal_edges if edge[-1] == key], dtype=np.float32)}
        for point in self.metal_nodes:
            self.grid_arrays[f"M{int(point[-1])}"]["node"][int(point[0])][int(point[1])] = 1


    def get_routed_points(self, point, edges):
        edges_list = [list(edge) for edge in edges]
        routed_points = np.array([point], dtype=np.float32)
        prv_routed_points = copy.deepcopy(routed_points)
        while True:
            copyed_routed_points = copy.deepcopy(routed_points)
            list_routed_points = [list(point) for point in routed_points]
            for p in copyed_routed_points:
                if list(p + np.array([0.5, 0, 0], dtype=np.float32)) in edges_list:
                    if list(p + np.array([1, 0, 0], dtype=np.float32)) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p + np.array([1, 0, 0], dtype=np.float32)], dtype=np.float32), axis=0)
                if list(p + np.array([0, 0.5, 0])) in edges_list:
                    if list(p + np.array([0, 1, 0], dtype=np.float32)) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p + np.array([0, 1, 0], dtype=np.float32)], dtype=np.float32), axis=0)
                if list(p + np.array([0, 0, 0.5], dtype=np.float32)) in edges_list:
                    if list(p + np.array([0, 0, 1], dtype=np.float32)) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p + np.array([0, 0, 1], dtype=np.float32)], dtype=np.float32), axis=0)
                if list(p - np.array([0.5, 0, 0], dtype=np.float32)) in edges_list:
                    if list(p - np.array([1, 0, 0], dtype=np.float32)) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p - np.array([1, 0, 0], dtype=np.float32)], dtype=np.float32), axis=0)
                if list(p - np.array([0, 0.5, 0], dtype=np.float32)) in edges_list:
                    if list(p - np.array([0, 1, 0], dtype=np.float32)) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p - np.array([0, 1, 0], dtype=np.float32)], dtype=np.float32), axis=0)
                if list(p - np.array([0, 0, 0.5], dtype=np.float32)) in edges_list:
                    if list(p - np.array([0, 0, 1], dtype=np.float32)) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p - np.array([0, 0, 1], dtype=np.float32)], dtype=np.float32), axis=0)
            # print(routed_points, prv_routed_points)
            # if np.all(routed_points == prv_routed_points): break
            if len(routed_points) == len(prv_routed_points): break
            else: prv_routed_points = copy.deepcopy(routed_points)

        return routed_points


    def update_start_goal_points(self, start_points, goal_points):
        start_point = start_points[0]; goal_point = goal_points[0]
        min_dist = np.linalg.norm(start_point - goal_point)
        for sp in self.start_points:
             for gp in self.goal_points:
                  if np.linalg.norm(sp - gp) <= min_dist:
                       start_point = sp; goal_point = gp; min_dist = np.linalg.norm(sp - gp)

        return start_point, goal_point


    def get_obs(self):
        # Vector to goal from moving point
        vec_to_goal = self.goal_point - self.moving_point

        # Return Observation (normalized : (-1~1))
        obs = np.array([self.cur_layer_bool,
                        vec_to_goal[0]/(self.max_x-self.min_x), vec_to_goal[1]/(self.max_y-self.min_y)], dtype=np.float32)
        near_objects = np.array([], dtype=np.float32)
        # for z in range(-self.obs_search_dist, self.obs_search_dist + 1):
        for z in range(-1, 2):
            for y in range(-self.obs_search_dist+abs(z), self.obs_search_dist-abs(z)+1):
                for x in range(-self.obs_search_dist+abs(z)+abs(y), self.obs_search_dist-abs(z)-abs(y)+1):
                    if list(self.moving_point + np.array([x, y, z], dtype=np.float32)) in [list(goal) for goal in self.goal_points]:
                        # goal points
                        appending_int = np.float32(1.0)
                    elif list(self.moving_point + np.array([x, y, z], dtype=np.float32)) in [list(node) for node in self.metal_nodes]:
                        # obstacle points
                        appending_int = np.float32(-1.0)
                    else:
                        # start points & empty points
                        appending_int = np.float32(0.0)
                    near_objects = np.append(near_objects, appending_int)
        obs = np.append(obs, near_objects)
        return obs


    def check_termination(self):
        # if moving point meets the obstacle : obstacle reached
        self.obstacle_reached = bool(list(self.moving_point) in [list(el) for el in self.metal_nodes]\
                                    or list(self.prv_point) == list(self.moving_point))
        
        if list(self.prv_point) in [list(el) for el in self.metal_nodes] and list(self.moving_point) in [list(el) for el in self.metal_nodes]:
            if list(np.mean([self.prv_point, self.moving_point], axis=0)) in [list(el) for el in self.metal_edges]:
                    self.obstacle_reached = False

        # when the agent goes again to the point in the trajectory
        if list(self.moving_point) in [list(traj) for traj in self.trajectory[:-1]]:
            self.obstacle_reached = True

        if self.moving_point[0] > self.max_x or self.moving_point[0] < self.min_x\
            or self.moving_point[1] > self.max_y or self.moving_point[1] < self.min_y:
                self.obstacle_reached = True

        # if moving point goes to goal point : goal reached
        self.goal_reached = bool(list(self.moving_point) in [list(goal) for goal in self.goal_points])
        if self.goal_reached: self.obstacle_reached = False

        # print(self.moving_point)
        # print(self.obstacle_reached)
        # print(self.goal_reached)


    def get_rwd(self, action):

        start_goal_dist = min(np.linalg.norm(goal_point - self.start_point) for goal_point in self.goal_points)
        agent_goal_dist = min(np.linalg.norm(goal_point - self.moving_point) for goal_point in self.goal_points)
        prv_goal_dist = min(np.linalg.norm(goal_point - self.prv_point) for goal_point in self.goal_points)

        reward = - (agent_goal_dist - prv_goal_dist)

        if self.obstacle_reached:
            reward = - start_goal_dist

        if list(np.mean([self.prv_prv_point, self.moving_point], axis=0)) != list(self.prv_point):
            reward += 0.5*(agent_goal_dist - prv_goal_dist)

        return reward


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent
        self.moving_point   = copy.deepcopy(self.start_point)
        self.prv_point      = copy.deepcopy(self.moving_point)
        self.prv_prv_point  = copy.deepcopy(self.prv_point)
        self.trajectory     = np.array([self.start_point], dtype=np.float32)
        self.via_tag        = [False]
        
        self.cur_layer_bool = 1 if self.moving_point[-1] % 2 == 0 else -1

        obs = self.get_obs()

        return obs


    def step(self, action):
        # 0 : +[1, 0, 0], 1 : -[1, 0, 0], 2 : +[0, 0, 1], 3 : -[0, 0, 1]

        # moving point
        if action == 0:
            if self.cur_layer_bool == 1:
                self.moving_point += np.array([1, 0, 0], dtype=np.float32)
            elif self.cur_layer_bool == -1:
                self.moving_point += np.array([0, 1, 0], dtype=np.float32)
        elif action == 1:
            if self.cur_layer_bool == 1:
                self.moving_point += np.array([-1, 0, 0], dtype=np.float32)
            elif self.cur_layer_bool == -1:
                self.moving_point += np.array([0, -1, 0], dtype=np.float32)
        elif action == 2:
            self.moving_point += np.array([0, 0, 1], dtype=np.float32)
        elif action == 3:
            self.moving_point += np.array([0, 0, -1], dtype=np.float32)

        # trajectory
        self.trajectory = np.append(self.trajectory, np.array([self.moving_point], dtype=np.float32), axis=0)
        # via_tag
        if self.prv_point[-1] != self.moving_point[-1]:
            self.via_tag.pop()
            self.via_tag.append(True)
        else:
            self.via_tag.append(False)

        # observation
        obs = self.get_obs()

        # termination condition
        self.check_termination()

        # REWARD
        reward = float(self.get_rwd(action))

        # done
        done = bool(self.goal_reached or self.obstacle_reached)

        # Optionally we can pass additional info, we are not using that for now
        info = {"trajectory": self.trajectory, "via_tag": self.via_tag}

        self.prv_prv_point = copy.deepcopy(self.prv_point)
        self.prv_point = copy.deepcopy(self.moving_point)
        self.cur_layer_bool = 1 if self.moving_point[-1] % 2 == 0 else -1
        # print(reward)
        # print()
        return obs, reward, done, info


    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(len(self.grid_arrays["M3"]["edge"]))
            for key, value in self.grid_arrays.items():
                print(f"{key} table")
                print(f"moving metal : {int(self.moving_point[-1])}")
                for j in range(int(self.max_y), int(self.min_y-1), -1):
                    for i in range(int(self.min_x), int(self.max_x+1)):
                        if [i, j] == list(self.moving_point[:-1]):
                            print("@", end="")
                            if len(value["edge"] > 0):
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                            else:                                
                                if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                    and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                        print("*", end="")
                                else:
                                    print(" ", end="")
                        elif [i, j] == list(self.start_point[:-1]):
                            print("S", end="")
                            if len(value["edge"] > 0):
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                            else:
                                if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                    and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                        print("*", end="")
                                else:
                                    print(" ", end="")
                        elif [i, j] == list(self.goal_point[:-1]):
                            print("E", end="")
                            if len(value["edge"] > 0):
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                            else:
                                if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                    and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                        print("*", end="")
                                else:
                                    print(" ", end="")
                        elif [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                            print(int(self.trajectory[[list(traj[:-1]) for traj in self.trajectory].index([i, j])][-1]), end="")
                            if len(value["edge"] > 0):
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                            else:                                
                                if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                    and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                        print("*", end="")
                                else:
                                    print(" ", end="")
                        elif value["node"][i][j] == 1:
                            print("X", end="")
                            if len(value["edge"] > 0):
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                            else:                                
                                if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                    and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                        print("*", end="")
                                else:
                                    print(" ", end="")
                        else: print("O", end=" ")
                    print()
                    for i in range(int(self.min_x), int(self.max_x+1)):
                        if len(value["edge"] > 0):
                            for p in value["edge"]:
                                if list(p) != []:
                                    if p[0] == i and p[1] == j-0.5:
                                        print("|", end=""); break
                                    elif p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]:
                                        if [i, j-1] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                        else:
                            if [i, j-1] in [list(traj[:-1]) for traj in self.trajectory]\
                                and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                    print("*", end="")
                            else:
                                print(" ", end="")
                        print("", end=" ")
                    print()
                print()
        print("goal reached     : ", self.goal_reached)
        print("obstacle reached : ", self.obstacle_reached)
        print("start point : ", self.start_point)
        print(self.trajectory)
        print("goal point  : ", self.goal_point)

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
        action = np.random.randint(3)
        print(action)
        obs, rewards, done, info = env.step(action)
        print(info)
        env.render()

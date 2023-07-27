import numpy as np
import gym
from gym import spaces
import copy


class RoutingEnv(gym.Env):
    def __init__(self,
                 init_metal_nodes   =   np.array([[2, 3, 1], [2, 5, 1], [6, 3, 1], [6, 5, 1],\
                                                  [1, 1, 1], [1, 2, 1], [3, 1, 1], [3, 2, 1], [5, 1, 1], [5, 2, 1], [7, 1, 1], [7, 2, 1],\
                                                  [1, 1, 2], [2, 1, 2], [3, 1, 2], [1, 3, 2], [2, 3, 2], [3, 3, 2], [5, 3, 2], [6, 3, 2], [7, 3, 2],\
                                                  [0, 0, 2], [1, 0, 2], [2, 0, 2], [3, 0, 2], [4, 0, 2], [5, 0, 2], [6, 0, 2], [7, 0, 2], [8, 0, 2], [9, 0, 2], [10, 0, 2], [11, 0, 2],\
                                                  [0, 8, 2], [1, 8, 2], [2, 8, 2], [3, 8, 2], [4, 8, 2], [5, 8, 2], [6, 8, 2], [7, 8, 2], [8, 8, 2], [9, 8, 2], [10, 8, 2], [11, 8, 2]]),
                 init_metal_edges   =   np.array([[1, 1.5, 1], [3, 1.5, 1], [5, 1.5, 1], [7, 1.5, 1],
                                                  [1.5, 1, 2], [2.5, 1, 2], [1.5, 3, 2], [2.5, 3, 2], [5.5, 3, 2], [6.5, 3, 2],
                                                  [0.5, 0, 2], [1.5, 0, 2], [2.5, 0, 2], [3.5, 0, 2], [4.5, 0, 2], [5.5, 0, 2], [6.5, 0, 2], [7.5, 0, 2], [8.5, 0, 2], [9.5, 0, 2], [10.5, 0, 2],\
                                                  [0.5, 8, 2], [1.5, 8, 2], [2.5, 8, 2], [3.5, 8, 2], [4.5, 8, 2], [5.5, 8, 2], [6.5, 8, 2], [7.5, 8, 2], [8.5, 8, 2], [9.5, 8, 2], [10.5, 8, 2]]),
                 points_to_route    =   {"A": np.array([[8, 3, 2], [10, 5, 2]]),
                                         "B": np.array([[2, 3, 2], [4, 5, 2]]),
                                         "internal": np.array([[2, 2, 2], [7, 1, 2]]),
                                         "O": np.array([[8, 2, 2], [8, 6, 2], [2, 6, 2]])}
                 routing_grid       =   "12",
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
        
        self.lower_layer            =   int(routing_grid[0])
        self.upper_layer            =   int(routing_grid[1])
        self.render_mode            =   render_mode

        self.moving_point           =   copy.deepcopy(self.start_point)
        self.moving_metal           =   self.moving_point[-1]
        self.prv_point              =   copy.deepcopy(self.moving_point)
        self.prv_metal              =   copy.deepcopy(self.moving_metal)

        self.trajectory             =   np.array([self.start_point])
        self.via_tag                =   [False]

        self.goal_reached           =   False
        self.obstacle_reached       =   False


        self.max_x, self.max_y = start_point[0], start_point[1]
        self.min_x, self.min_y = start_point[0], start_point[1]
        for point in self.metal_nodes:
            if point[0] >= self.max_x: self.max_x = point[0]
            if point[1] >= self.max_y: self.max_y = point[1]
            if point[0] <= self.min_x: self.min_x = point[0]
            if point[1] <= self.min_y: self.min_y = point[1]
        
        self.update_grid_arrays()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(3)   # 0 : go front, 1 : go back, 2 : metal change
        self.observation_space = spaces.Box(
            low=-max(self.max_x, self.max_y), high=max(self.max_x, self.max_y), shape=(13,), dtype=np.float32
        )

    def update_grid_arrays(self):
        self.grid_array             =   np.zeros((self.max_x - self.min_x + 1, self.max_y - self.min_y + 1))
        self.grid_shape             =   np.shape(self.grid_array)
        self.grid_arrays            =   dict()
        for key in range(min([p[-1] for p in self.metal_nodes]), max([p[-1] for p in self.metal_nodes]) + 1):
            self.grid_arrays[f"M{key}"] = {"node": copy.deepcopy(self.grid_array),\
                                           "edge": np.array([edge[:-1] for edge in self.metal_edges if edge[-1] == key])}
        for point in self.metal_nodes:
            self.grid_arrays[f"M{point[-1]}"]["node"][point[0]][point[1]] = 1

        if f"M{self.lower_layer}" not in self.grid_arrays.keys():
            self.grid_arrays[f"M{self.lower_layer}"] = {"node": copy.deepcopy(self.grid_array),\
                                                        "edge": np.array([[]])}
        if f"M{self.upper_layer}" not in self.grid_arrays.keys():
            self.grid_arrays[f"M{self.upper_layer}"] = {"node": copy.deepcopy(self.grid_array),\
                                                        "edge": np.array([[]])}


    def get_routed_points(self, point, edges):
        edges_list = [list(edge) for edge in edges]
        routed_points = np.array([point])
        prv_routed_points = copy.deepcopy(routed_points)
        while True:
            copyed_routed_points = copy.deepcopy(routed_points)
            list_routed_points = [list(point) for point in routed_points]
            for p in copyed_routed_points:
                if list(p + np.array([0.5, 0, 0])) in edges_list:
                    if list(p + np.array([1, 0, 0])) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p + np.array([1, 0, 0])]), axis=0)
                if list(p + np.array([0, 0.5, 0])) in edges_list:
                    if list(p + np.array([0, 1, 0])) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p + np.array([0, 1, 0])]), axis=0)
                if list(p + np.array([0, 0, 0.5])) in edges_list:
                    if list(p + np.array([0, 0, 1])) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p + np.array([0, 0, 1])]), axis=0)
                if list(p - np.array([0.5, 0, 0])) in edges_list:
                    if list(p - np.array([1, 0, 0])) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p - np.array([1, 0, 0])]), axis=0)
                if list(p - np.array([0, 0.5, 0])) in edges_list:
                    if list(p - np.array([0, 1, 0])) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p - np.array([0, 1, 0])]), axis=0)
                if list(p - np.array([0, 0, 0.5])) in edges_list:
                    if list(p - np.array([0, 0, 1])) not in list_routed_points:
                        routed_points = np.append(routed_points, np.array([p - np.array([0, 0, 1])]), axis=0)
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
        vec_to_start = self.start_point - self.moving_point
        start_to_goal = self.goal_point - self.start_point
        # Vectors to nearest obstacles from current/non-current layer metals

        # M1, M3, ...
        if self.moving_metal % 2 == 1:
            # current layer nearest obstacle
            vec_moving_layer_obstacle_goal_dir = np.array([0, vec_to_goal[1]])
            vec_moving_layer_obstacle_start_dir = np.array([0, vec_to_start[1]])
            vec_nonmoving_layer_obstacle_goal_dir = np.array([vec_to_goal[0], 0])
            vec_nonmoving_layer_obstacle_start_dir = np.array([vec_to_start[0], 0])

            for y in range(0, vec_to_goal[1], 1 if vec_to_goal[1] >= 0 else -1):
                if f"M{self.moving_metal}" in self.grid_arrays.keys():
                    if self.grid_arrays[f"M{self.moving_metal}"]["node"]\
                        [self.moving_point[0]][self.moving_point[1] + y] == 1:
                            if list(self.moving_point + np.array([0, y, 0])) == list(self.goal_point):
                                vec_moving_layer_obstacle_goal_dir = np.array([0, abs(start_to_goal)+1]); break
                            if y != 0:
                                search_point = self.moving_point + np.array([0, y, 0])
                                search_point_prv = self.moving_point + np.array([0, y - y/abs(y), 0])
                                if np.mean([search_point, search_point_prv], axis=0) in self.metal_edges\
                                    or list(search_point) in [list(goal) for goal in self.goal_points]:
                                        continue
                                vec_moving_layer_obstacle_goal_dir = np.array([0, y]); break
            for y in range(0, vec_to_start[1], 1 if vec_to_start[1] >= 0 else -1):
                if f"M{self.moving_metal}" in self.grid_arrays.keys():
                    if self.grid_arrays[f"M{self.moving_metal}"]["node"]\
                        [self.moving_point[0]][self.moving_point[1] + y] == 1:
                            if y != 0:
                                search_point = self.moving_point + np.array([0, y, 0])
                                search_point_prv = self.moving_point + np.array([0, y - y/abs(y), 0])
                                if np.mean([search_point, search_point_prv], axis=0) in self.metal_edges\
                                    or list(search_point) in [list(goal) for goal in self.goal_points]:
                                        continue
                                vec_moving_layer_obstacle_start_dir = np.array([0, y]); break

            # currently in the lower metal of a routing grid
            if self.moving_metal == self.lower_layer:
                # upper layer nearest obstacle
                for x_ in range(0, vec_to_goal[0], 1 if vec_to_goal[0] >= 0 else -1):
                    if f"M{self.moving_metal+1}" in self.grid_arrays.keys():
                        if self.grid_arrays[f"M{self.moving_metal+1}"]["node"]\
                            [self.moving_point[0] + x_][self.moving_point[1]] == 1:
                                vec_nonmoving_layer_obstacle_goal_dir = np.array([x_, 0]); break
                for x_ in range(0, vec_to_start[0], 1 if vec_to_start[0] >= 0 else -1):
                    if f"M{self.moving_metal+1}" in self.grid_arrays.keys():
                        if self.grid_arrays[f"M{self.moving_metal+1}"]["node"]\
                            [self.moving_point[0] + x_][self.moving_point[1]] == 1:
                                vec_nonmoving_layer_obstacle_start_dir = np.array([x_, 0]); break
            # currently in the upper metal of a routing grid
            elif self.moving_metal == self.upper_layer:
                # lower layer nearest obstacle
                for x_ in range(0, vec_to_goal[0], 1 if vec_to_goal[0] >= 0 else -1):
                    if f"M{self.moving_metal-1}" in self.grid_arrays.keys():
                        if self.grid_arrays[f"M{self.moving_metal-1}"]["node"]\
                            [self.moving_point[0] + x_][self.moving_point[1]] == 1:
                                vec_nonmoving_layer_obstacle_goal_dir = np.array([x_, 0]); break
                for x_ in range(0, vec_to_start[0], 1 if vec_to_start[0] >= 0 else -1):
                    if f"M{self.moving_metal-1}" in self.grid_arrays.keys():
                        if self.grid_arrays[f"M{self.moving_metal-1}"]["node"]\
                            [self.moving_point[0] + x_][self.moving_point[1]] == 1:
                                vec_nonmoving_layer_obstacle_start_dir = np.array([x_, 0]); break


        # M2, M4, ...
        elif self.moving_metal % 2 == 0:
            # current layer nearest obstacle
            vec_moving_layer_obstacle_goal_dir = np.array([vec_to_goal[0], 0])
            vec_moving_layer_obstacle_start_dir = np.array([vec_to_start[0], 0])
            vec_nonmoving_layer_obstacle_goal_dir = np.array([0, vec_to_goal[1]])
            vec_nonmoving_layer_obstacle_start_dir = np.array([0, vec_to_start[1]])

            for x in range(0, vec_to_goal[0], 1 if vec_to_goal[0] >= 0 else -1):
                if f"M{self.moving_metal}" in self.grid_arrays.keys():
                    if self.grid_arrays[f"M{self.moving_metal}"]["node"]\
                        [self.moving_point[0] + x][self.moving_point[1]] == 1:
                            if list(self.moving_point + np.array([x, 0, 0])) == list(self.goal_point):
                                vec_moving_layer_obstacle_goal_dir = np.array([abs(start_to_goal)+1, 0]); break
                            if x != 0:
                                search_point = self.moving_point + np.array([x, 0, 0])
                                search_point_prv = self.moving_point + np.array([x - x/abs(x), 0, 0])
                                if np.mean([search_point, search_point_prv], axis=0) in self.metal_edges\
                                    or list(search_point) in [list(goal) for goal in self.goal_points]:
                                        continue
                                vec_moving_layer_obstacle_goal_dir = np.array([x, 0]); break
            for x in range(0, vec_to_start[0], 1 if vec_to_start[0] >= 0 else -1):
                if f"M{self.moving_metal}" in self.grid_arrays.keys():
                    if self.grid_arrays[f"M{self.moving_metal}"]["node"]\
                        [self.moving_point[0] + x][self.moving_point[1]] == 1:
                            if x != 0:
                                search_point = self.moving_point + np.array([x, 0, 0])
                                search_point_prv = self.moving_point + np.array([x - x/abs(x), 0, 0])
                                if np.mean([search_point, search_point_prv], axis=0) in self.metal_edges\
                                    or list(search_point) in [list(goal) for goal in self.goal_points]:
                                        continue
                                vec_moving_layer_obstacle_start_dir = np.array([x, 0]); break

            # currently in the lower metal of a routing grid
            if self.moving_metal == self.lower_layer:
                # upper layer nearest obstacle
                for y_ in range(0, vec_to_goal[1], 1 if vec_to_goal[1] >= 0 else -1):
                    if f"M{self.moving_metal+1}" in self.grid_arrays.keys():
                        if self.grid_arrays[f"M{self.moving_metal+1}"]["node"]\
                            [self.moving_point[0]][self.moving_point[1] + y_] == 1:
                                vec_nonmoving_layer_obstacle_goal_dir = np.array([0, y_]); break
                for y_ in range(0, vec_to_start[1], 1 if vec_to_start[1] >= 0 else -1):
                    if f"M{self.moving_metal+1}" in self.grid_arrays.keys():
                        if self.grid_arrays[f"M{self.moving_metal+1}"]["node"]\
                            [self.moving_point[0]][self.moving_point[1] + y_] == 1:
                                vec_nonmoving_layer_obstacle_start_dir = np.array([0, y_]); break
            # currently in the upper metal of a routing grid
            elif self.moving_metal == self.upper_layer:
                # lower layer nearest obstacle
                for y_ in range(0, vec_to_goal[1], 1 if vec_to_goal[1] >= 0 else -1):
                    if f"M{self.moving_metal-1}" in self.grid_arrays.keys():
                        if self.grid_arrays[f"M{self.moving_metal-1}"]["node"]\
                            [self.moving_point[0]][self.moving_point[1] + y_] == 1:
                                vec_nonmoving_layer_obstacle_goal_dir = np.array([0, y_]); break
                for y_ in range(0, vec_to_start[1], 1 if vec_to_start[1] >= 0 else -1):
                    if f"M{self.moving_metal-1}" in self.grid_arrays.keys():
                        if self.grid_arrays[f"M{self.moving_metal-1}"]["node"]\
                            [self.moving_point[0]][self.moving_point[1] + y_] == 1:
                                vec_nonmoving_layer_obstacle_start_dir = np.array([0, y_]); break

        # print(self.min_x, self.max_x)
        # print(self.min_y, self.max_y)

        # Return Observation (normalize)
        obs = np.array([(self.moving_point[0]-self.min_x)/(self.max_x-self.min_x), (self.moving_point[1]-self.min_y)/(self.max_y-self.min_y),\
                        self.moving_metal/max([node[-1] for node in self.metal_nodes]),\
                        vec_to_goal[0]/(abs(start_to_goal[0])+1), vec_to_goal[1]/(abs(start_to_goal[1])+1),\
                        vec_moving_layer_obstacle_goal_dir[0]/(abs(start_to_goal[0])+1), vec_moving_layer_obstacle_goal_dir[1]/(abs(start_to_goal[1])+1),\
                        vec_moving_layer_obstacle_start_dir[0]/(abs(start_to_goal[0])+1), vec_moving_layer_obstacle_start_dir[1]/(abs(start_to_goal[1])+1),\
                        vec_nonmoving_layer_obstacle_goal_dir[0]/(abs(start_to_goal[0])+1), vec_nonmoving_layer_obstacle_goal_dir[1]/(abs(start_to_goal[1])+1),\
                        vec_nonmoving_layer_obstacle_start_dir[0]/(abs(start_to_goal[0])+1), vec_nonmoving_layer_obstacle_start_dir[1]/(abs(start_to_goal[1])+1)], dtype=np.float32)
        
        # print(obs)

        return obs


    def check_termination(self):
        # if moving point meets the obstacle : obstacle reached
        self.obstacle_reached = bool(list(self.moving_point) in [list(el) for el in self.metal_nodes]\
                                    or list(self.prv_point) == list(self.moving_point))
        
        if list(self.prv_point) in [list(el) for el in self.metal_nodes] and list(self.moving_point) in [list(el) for el in self.metal_nodes]:
            if list(np.mean([self.prv_point, self.moving_point], axis=0)) in [list(el) for el in self.metal_edges]\
                or list(self.prv_point) == list(self.moving_point):
                    self.obstacle_reached = False

        # when the agent goes again to the point in the trajectory
        if list(self.moving_point) in [list(traj) for traj in self.trajectory[:-1]]:
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

        # if list(self.moving_point) in [list(point) for point in self.start_points]:
        #      reward += 0.1

        if self.obstacle_reached:
            reward = - start_goal_dist

        return reward


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent
        self.moving_point = copy.deepcopy(self.start_point)
        self.moving_metal = self.lower_layer
        self.trajectory   = np.array([self.start_point])
        self.via_tag      = [False]
        self.prv_point    = copy.deepcopy(self.moving_point)
        self.prv_metal    = self.moving_metal
        
        obs = self.get_obs()
        # print("start")
        return obs  # empty info dict


    def step(self, action):
        # action == 0:   # + 1
        # action == 1:   # - 1
        # action == 2:   # metal change
        
        # print(self.moving_point)

        # moving_metal
        if action == 2:
            if self.moving_metal == self.lower_layer:
                self.moving_metal = self.upper_layer
            elif self.moving_metal == self.upper_layer:
                self.moving_metal = self.lower_layer
            self.moving_point[-1] = self.moving_metal
        
        # moving point
        if self.moving_metal % 2 == 1:      # M1, M3, ...
            if action == 0:     # + 1
                self.moving_point += np.array([0, 1, 0])
            elif action == 1:   # - 1
                self.moving_point += np.array([0, -1, 0])
            elif action == 2:   # metal change
                pass
        elif self.moving_metal % 2 == 0:    # M2, M4, ...
            if action == 0:     # + 1
                self.moving_point += np.array([1, 0, 0])
            elif action == 1:   # - 1
                self.moving_point += np.array([-1, 0, 0])
            elif action == 2:   # metal change
                pass

        # Account for the boundaries of the grid
        if self.moving_point[0] > self.max_x: self.moving_point[0] = self.max_x
        if self.moving_point[1] > self.max_y: self.moving_point[1] = self.max_y
        if self.moving_point[0] < self.min_x: self.moving_point[0] = self.min_x
        if self.moving_point[1] < self.min_y: self.moving_point[1] = self.min_y

        # trajectory
        self.trajectory = np.append(self.trajectory, np.array([self.moving_point]), axis=0)
        # via_tag
        if self.prv_metal != self.moving_metal:
            self.via_tag.pop()
            self.via_tag.append(True)
        else:
            self.via_tag.append(False)

        # observation
        obs = self.get_obs()

        # termination condition
        self.check_termination()

        # REWARD
        reward = self.get_rwd(action)

        # done
        done = bool(self.goal_reached or self.obstacle_reached)

        # Optionally we can pass additional info, we are not using that for now
        info = {"trajectory": self.trajectory, "via_tag": self.via_tag}

        self.prv_point = copy.deepcopy(self.moving_point)
        self.prv_metal = self.moving_metal
        # print(reward)
        # print()
        return obs, reward, done, info


    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            for key, value in self.grid_arrays.items():
                if int(key[-1]) in [self.lower_layer, self.upper_layer]:
                    print(f"{key} table")
                    print(f"moving metal : {self.moving_metal}")
                    for j in range(self.max_y, self.min_y-1, -1):
                        for i in range(self.min_x, self.max_x+1):
                            if [i, j] == list(self.moving_point[:-1]):
                                print("@", end="")
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                            elif [i, j] == list(self.start_point[:-1]):
                                print("S", end="")
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break                         
                            elif [i, j] == list(self.goal_point[:-1]):
                                print("E", end="")
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                            elif [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                print(self.trajectory[[list(traj[:-1]) for traj in self.trajectory].index([i, j])][-1], end="")
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                            elif value["node"][i][j] == 1:
                                print("X", end="")
                                for p in value["edge"]:
                                    if list(p) != [] and p[0] == i+0.5 and p[1] == j:
                                        print("-", end=""); break
                                    elif list(p) == [] or (p[0] == value["edge"][-1][0] and p[1] == value["edge"][-1][1]):
                                        if [i+1, j] in [list(traj[:-1]) for traj in self.trajectory]\
                                            and [i, j] in [list(traj[:-1]) for traj in self.trajectory]:
                                                print("*", end=""); break
                                        else:
                                            print(" ", end=""); break
                            else: print("O", end=" ")
                        print()
                        for i in range(self.min_x, self.max_x+1):
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

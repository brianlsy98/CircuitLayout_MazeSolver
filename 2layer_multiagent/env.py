import numpy as np
import gym
from gym import spaces
import copy


def get_routed_points(point, edges):
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

def update_start_goal_points(start_points, goal_points):
    start_point = start_points[0]; goal_point = goal_points[0]
    min_dist = np.linalg.norm(start_point[:-1] - goal_point[:-1])
    for sp in start_points:
            for gp in goal_points:
                if np.linalg.norm(sp[:-1] - gp[:-1]) <= min_dist:
                    start_point = sp; goal_point = gp; min_dist = np.linalg.norm(sp[:-1] - gp[:-1])

    return start_point, goal_point



class Agent():
    def __init__(self, name,
                 metal_nodes, metal_edges,
                 start_point, end_point,
                 max_x, max_y, min_x, min_y,
                 upper_layer, lower_layer, render_mode):
        self.name = name
        self.metal_nodes = copy.deepcopy(metal_nodes)
        self.metal_edges = copy.deepcopy(metal_edges)
        self.start_point = copy.deepcopy(start_point)
        self.start_points = get_routed_points(self.start_point, self.metal_edges)
        self.goal_point = copy.deepcopy(end_point)
        self.goal_points = get_routed_points(self.goal_point, self.metal_edges)
        self.start_point, self.goal_point\
                                    =   update_start_goal_points(self.start_points, self.goal_points)

        self.moving_point           =   copy.deepcopy(self.start_point)
        self.prv_point              =   copy.deepcopy(self.moving_point)
        self.cur_layer_bool         =   1 if self.moving_point[-1] % 2 == 0 else -1
        self.upper_layer            =   upper_layer
        self.lower_layer            =   lower_layer

        self.trajectory             =   np.array([self.start_point])
        self.via_tag                =   [False]

        self.goal_reached           =   False
        self.obstacle_reached       =   False

        self.max_x, self.max_y = max_x, max_y
        self.min_x, self.min_y = min_x, min_y

        self.update_grid_arrays()

        self.render_mode            =   render_mode

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

    def get_obs_lidarlike(self):
        # Vector to goal from moving point
        vec_to_goal = self.goal_point - self.moving_point

        # Return Observation (normalized : (-1~1))
        obs = np.array([self.cur_layer_bool,
                        vec_to_goal[0]/(self.max_x-self.min_x), vec_to_goal[1]/(self.max_y-self.min_y)], dtype=np.float32)
        
        near_objects = np.array([], dtype=np.float32)
        if self.lower_layer % 2 == 0: # M2, M4, ...
            z = 0 if self.moving_point[-1] == self.lower_layer else -1

            # lower layer half1
            for i in range(1, 4):
                if list(self.moving_point + np.array([i, i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, i, z])) not in [list(goal) for goal in self.goal_points]:
                        a0 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([i, i, z])) in [list(goal) for goal in self.goal_points]:
                    a0 = np.float32(1 - (i-1)/3); break
                else: a0 = np.float32(0)
            near_objects = np.append(near_objects, a0)
            for i in range(1, 3):
                if list(self.moving_point + np.array([2*i, i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([2*i, i, z])) not in [list(goal) for goal in self.goal_points]:
                        a1 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([2*i, i, z])) in [list(goal) for goal in self.goal_points]:
                    a1 = np.float32(1 - (i-1)/2); break
                else: a1 = np.float32(0)
            near_objects = np.append(near_objects, a1)
            for i in range(1, 5):
                if list(self.moving_point + np.array([i, 0, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, 0, z])) not in [list(goal) for goal in self.goal_points]:
                        a2 = np.float32(-1 + (i-1)/4); break
                elif list(self.moving_point + np.array([i, 0, z])) in [list(goal) for goal in self.goal_points]:
                    a2 = np.float32(1 - (i-1)/4); break
                else: a2 = np.float32(0)
            near_objects = np.append(near_objects, a2)
            for i in range(1, 3):
                if list(self.moving_point + np.array([2*i, -i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([2*i, -i, z])) not in [list(goal) for goal in self.goal_points]:
                        a3 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([2*i, -i, z])) in [list(goal) for goal in self.goal_points]:
                    a3 = np.float32(1 - (i-1)/2); break
                else: a3 = np.float32(0)
            near_objects = np.append(near_objects, a3)
            for i in range(1, 4):
                if list(self.moving_point + np.array([i, -i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, -i, z])) not in [list(goal) for goal in self.goal_points]:
                        a4 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([i, -i, z])) in [list(goal) for goal in self.goal_points]:
                    a4 = np.float32(1 - (i-1)/3); break
                else: a4 = np.float32(0)
            near_objects = np.append(near_objects, a4)
            # upper layer half1
            for i in range(1, 4):
                if list(self.moving_point + np.array([i, -i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, -i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a5 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([i, -i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a5 = np.float32(1 - (i-1)/3); break
                else: a5 = np.float32(0)
            near_objects = np.append(near_objects, a5)
            for i in range(1, 3):
                if list(self.moving_point + np.array([i, -2*i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, -2*i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a6 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([i, -2*i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a6 = np.float32(1 - (i-1)/2); break
                else: a6 = np.float32(0)
            near_objects = np.append(near_objects, a6)
            for i in range(1, 5):
                if list(self.moving_point + np.array([0, -i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([0, -i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a7 = np.float32(-1 + (i-1)/4); break
                elif list(self.moving_point + np.array([0, -i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a7 = np.float32(1 - (i-1)/4); break
                else: a7 = np.float32(0)
            near_objects = np.append(near_objects, a7)
            for i in range(1, 3):
                if list(self.moving_point + np.array([-i, -2*i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, -2*i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a8 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([-i, -2*i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a8 = np.float32(1 - (i-1)/2); break
                else: a8 = np.float32(0)
            near_objects = np.append(near_objects, a8)
            for i in range(1, 4):
                if list(self.moving_point + np.array([-i, -i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, -i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a9 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([-i, -i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a9 = np.float32(1 - (i-1)/3); break
                else: a9 = np.float32(0)
            near_objects = np.append(near_objects, a9)
            # lower layer half2
            for i in range(1, 4):
                if list(self.moving_point + np.array([-i, -i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, -i, z])) not in [list(goal) for goal in self.goal_points]:
                        a10 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([-i, -i, z])) in [list(goal) for goal in self.goal_points]:
                    a10 = np.float32(1 - (i-1)/3); break
                else: a10 = np.float32(0)
            near_objects = np.append(near_objects, a10)
            for i in range(1, 3):
                if list(self.moving_point + np.array([-2*i, -i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-2*i, -i, z])) not in [list(goal) for goal in self.goal_points]:
                        a11 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([-2*i, -i, z])) in [list(goal) for goal in self.goal_points]:
                    a11 = np.float32(1 - (i-1)/2); break
                else: a11 = np.float32(0)
            near_objects = np.append(near_objects, a11)
            for i in range(1, 5):
                if list(self.moving_point + np.array([-i, 0, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, 0, z])) not in [list(goal) for goal in self.goal_points]:
                        a12 = np.float32(-1 + (i-1)/4); break
                elif list(self.moving_point + np.array([-i, 0, z])) in [list(goal) for goal in self.goal_points]:
                    a12 = np.float32(1 - (i-1)/4); break
                else: a12 = np.float32(0)
            near_objects = np.append(near_objects, a12)
            for i in range(1, 3):
                if list(self.moving_point + np.array([-2*i, i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-2*i, i, z])) not in [list(goal) for goal in self.goal_points]:
                        a13 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([-2*i, i, z])) in [list(goal) for goal in self.goal_points]:
                    a13 = np.float32(1 - (i-1)/2); break
                else: a13 = np.float32(0)
            near_objects = np.append(near_objects, a13)
            for i in range(1, 4):
                if list(self.moving_point + np.array([-i, i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, i, z])) not in [list(goal) for goal in self.goal_points]:
                        a14 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([-i, i, z])) in [list(goal) for goal in self.goal_points]:
                    a14 = np.float32(1 - (i-1)/3); break
                else: a14 = np.float32(0)
            near_objects = np.append(near_objects, a14)
            # upper layer half2
            for i in range(1, 4):
                if list(self.moving_point + np.array([-i, i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a15 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([-i, i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a15 = np.float32(1 - (i-1)/3); break
                else: a15 = np.float32(0)
            near_objects = np.append(near_objects, a15)
            for i in range(1, 3):
                if list(self.moving_point + np.array([-i, 2*i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, 2*i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a16 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([-i, 2*i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a16 = np.float32(1 - (i-1)/2); break
                else: a16 = np.float32(0)
            near_objects = np.append(near_objects, a16)
            for i in range(1, 5):
                if list(self.moving_point + np.array([0, i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([0, i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a17 = np.float32(-1 + (i-1)/4); break
                elif list(self.moving_point + np.array([0, i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a17 = np.float32(1 - (i-1)/4); break
                else: a17 = np.float32(0)
            near_objects = np.append(near_objects, a17)
            for i in range(1, 3):
                if list(self.moving_point + np.array([i, 2*i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, 2*i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a18 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([i, 2*i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a18 = np.float32(1 - (i-1)/2); break
                else: a18 = np.float32(0)
            near_objects = np.append(near_objects, a18)
            for i in range(1, 4):
                if list(self.moving_point + np.array([i, i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a19 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([i, i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a19 = np.float32(1 - (i-1)/3); break
                else: a19 = np.float32(0)
            near_objects = np.append(near_objects, a19)

        else: # lower layer is M1, M3, ...
            z = 0 if self.moving_point[-1] == self.lower_layer else -1

            # lower layer half1
            for i in range(1, 4):
                if list(self.moving_point + np.array([-i, i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, i, z])) not in [list(goal) for goal in self.goal_points]:
                        a0 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([-i, i, z])) in [list(goal) for goal in self.goal_points]:
                    a0 = np.float32(1 - (i-1)/3); break
                else: a0 = np.float32(0)
            near_objects = np.append(near_objects, a0)
            for i in range(1, 3):
                if list(self.moving_point + np.array([-i, 2*i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, 2*i, z])) not in [list(goal) for goal in self.goal_points]:
                        a1 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([-i, 2*i, z])) in [list(goal) for goal in self.goal_points]:
                    a1 = np.float32(1 - (i-1)/2); break
                else: a1 = np.float32(0)
            near_objects = np.append(near_objects, a1)
            for i in range(1, 5):
                if list(self.moving_point + np.array([0, i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([0, i, z])) not in [list(goal) for goal in self.goal_points]:
                        a2 = np.float32(-1 + (i-1)/4); break
                elif list(self.moving_point + np.array([0, i, z])) in [list(goal) for goal in self.goal_points]:
                    a2 = np.float32(1 - (i-1)/4); break
                else: a2 = np.float32(0)
            near_objects = np.append(near_objects, a2)
            for i in range(1, 3):
                if list(self.moving_point + np.array([i, 2*i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, 2*i, z])) not in [list(goal) for goal in self.goal_points]:
                        a3 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([i, 2*i, z])) in [list(goal) for goal in self.goal_points]:
                    a3 = np.float32(1 - (i-1)/2); break
                else: a3 = np.float32(0)
            near_objects = np.append(near_objects, a3)
            for i in range(1, 4):
                if list(self.moving_point + np.array([i, i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, i, z])) not in [list(goal) for goal in self.goal_points]:
                        a4 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([i, i, z])) in [list(goal) for goal in self.goal_points]:
                    a4 = np.float32(1 - (i-1)/3); break
                else: a4 = np.float32(0)
            near_objects = np.append(near_objects, a4)
            # upper layer half1
            for i in range(1, 4):
                if list(self.moving_point + np.array([i, i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a5 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([i, i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a5 = np.float32(1 - (i-1)/3); break
                else: a5 = np.float32(0)
            near_objects = np.append(near_objects, a5)
            for i in range(1, 3):
                if list(self.moving_point + np.array([2*i, i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([2*i, i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a6 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([2*i, i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a6 = np.float32(1 - (i-1)/2); break
                else: a6 = np.float32(0)
            near_objects = np.append(near_objects, a6)
            for i in range(1, 5):
                if list(self.moving_point + np.array([i, 0, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, 0, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a7 = np.float32(-1 + (i-1)/4); break
                elif list(self.moving_point + np.array([i, 0, z+1])) in [list(goal) for goal in self.goal_points]:
                    a7 = np.float32(1 - (i-1)/4); break
                else: a7 = np.float32(0)
            near_objects = np.append(near_objects, a7)
            for i in range(1, 3):
                if list(self.moving_point + np.array([2*i, -i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([2*i, -i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a8 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([2*i, -i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a8 = np.float32(1 - (i-1)/2); break
                else: a8 = np.float32(0)
            near_objects = np.append(near_objects, a8)
            for i in range(1, 4):
                if list(self.moving_point + np.array([i, -i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, -i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a9 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([i, -i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a9 = np.float32(1 - (i-1)/3); break
                else: a9 = np.float32(0)
            near_objects = np.append(near_objects, a9)
            # lower layer half2
            for i in range(1, 4):
                if list(self.moving_point + np.array([i, -i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, -i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a10 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([i, -i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a10 = np.float32(1 - (i-1)/3); break
                else: a10 = np.float32(0)
            near_objects = np.append(near_objects, a10)
            for i in range(1, 3):
                if list(self.moving_point + np.array([i, -2*i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([i, -2*i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a11 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([i, -2*i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a11 = np.float32(1 - (i-1)/2); break
                else: a11 = np.float32(0)
            near_objects = np.append(near_objects, a11)
            for i in range(1, 5):
                if list(self.moving_point + np.array([0, -i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([0, -i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a12 = np.float32(-1 + (i-1)/4); break
                elif list(self.moving_point + np.array([0, -i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a12 = np.float32(1 - (i-1)/4); break
                else: a12 = np.float32(0)
            near_objects = np.append(near_objects, a12)
            for i in range(1, 3):
                if list(self.moving_point + np.array([-i, -2*i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, -2*i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a13 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([-i, -2*i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a13 = np.float32(1 - (i-1)/2); break
                else: a13 = np.float32(0)
            near_objects = np.append(near_objects, a13)
            for i in range(1, 4):
                if list(self.moving_point + np.array([-i, -i, z+1])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, -i, z+1])) not in [list(goal) for goal in self.goal_points]:
                        a14 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([-i, -i, z+1])) in [list(goal) for goal in self.goal_points]:
                    a14 = np.float32(1 - (i-1)/3); break
                else: a14 = np.float32(0)
            near_objects = np.append(near_objects, a14)
            # upper layer half2
            for i in range(1, 4):
                if list(self.moving_point + np.array([-i, -i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, -i, z])) not in [list(goal) for goal in self.goal_points]:
                        a15 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([-i, -i, z])) in [list(goal) for goal in self.goal_points]:
                    a15 = np.float32(1 - (i-1)/3); break
                else: a15 = np.float32(0)
            near_objects = np.append(near_objects, a15)
            for i in range(1, 3):
                if list(self.moving_point + np.array([-2*i, -i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-2*i, -i, z])) not in [list(goal) for goal in self.goal_points]:
                        a16 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([-2*i, -i, z])) in [list(goal) for goal in self.goal_points]:
                    a16 = np.float32(1 - (i-1)/2); break
                else: a16 = np.float32(0)
            near_objects = np.append(near_objects, a16)
            for i in range(1, 5):
                if list(self.moving_point + np.array([-i, 0, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, 0, z])) not in [list(goal) for goal in self.goal_points]:
                        a17 = np.float32(-1 + (i-1)/4); break
                elif list(self.moving_point + np.array([-i, 0, z])) in [list(goal) for goal in self.goal_points]:
                    a17 = np.float32(1 - (i-1)/4); break
                else: a17 = np.float32(0)
            near_objects = np.append(near_objects, a17)
            for i in range(1, 3):
                if list(self.moving_point + np.array([-2*i, i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-2*i, i, z])) not in [list(goal) for goal in self.goal_points]:
                        a18 = np.float32(-1 + (i-1)/2); break
                elif list(self.moving_point + np.array([-2*i, i, z])) in [list(goal) for goal in self.goal_points]:
                    a18 = np.float32(1 - (i-1)/2); break
                else: a18 = np.float32(0)
            near_objects = np.append(near_objects, a18)
            for i in range(1, 4):
                if list(self.moving_point + np.array([-i, i, z])) in [list(obstacle) for obstacle in self.metal_nodes]\
                    and list(self.moving_point + np.array([-i, i, z])) not in [list(goal) for goal in self.goal_points]:
                        a19 = np.float32(-1 + (i-1)/3); break
                elif list(self.moving_point + np.array([-i, i, z])) in [list(goal) for goal in self.goal_points]:
                    a19 = np.float32(1 - (i-1)/3); break
                else: a19 = np.float32(0)
            near_objects = np.append(near_objects, a19)

        obs = np.append(obs, near_objects)

        return obs

    def get_obs(self):
        # Vector to goal from moving point
        vec_to_goal = self.goal_point - self.moving_point

        # Return Observation (normalized : (-1~1))
        obs = np.array([self.cur_layer_bool,
                        vec_to_goal[0]/(self.max_x-self.min_x), vec_to_goal[1]/(self.max_y-self.min_y)], dtype=np.float32)
        near_objects = np.array([], dtype=np.float32)
        # for z in range(-self.obs_search_dist, self.obs_search_dist + 1):
        for z in range(-1, 2):
            for y in range(-2+abs(z), 2-abs(z)+1):
                for x in range(-2+abs(z)+abs(y), 2-abs(z)-abs(y)+1):
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

    def reset(self):
        self.moving_point           =   copy.deepcopy(self.start_point)
        self.prv_point              =   copy.deepcopy(self.moving_point)
        self.cur_layer_bool         =   1 if self.moving_point[-1] % 2 == 0 else -1
        self.trajectory             =   np.array([self.start_point])
        self.via_tag                =   [False]

        obs = self.get_obs()
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

    def step(self, action):
        # action[i] == 0:   # agent i : + 1
        # action[i] == 1:   # agent i : - 1
        # action[i] == 2:   # agent i : metal change

        if self.goal_reached: pass
        else:
            # moving point
            if self.moving_point[-1] % 2 == 1:      # M1, M3, ...
                if action == 0:     # + 1
                    self.moving_point += np.array([0, 1, 0])
                elif action == 1:   # - 1
                    self.moving_point += np.array([0, -1, 0])
            elif self.moving_point[-1] % 2 == 0:    # M2, M4, ...
                if action == 0:     # + 1
                    self.moving_point += np.array([1, 0, 0])
                elif action == 1:   # - 1
                    self.moving_point += np.array([-1, 0, 0])

            # moving_metal
            if action == 2:
                if self.moving_point[-1] == self.lower_layer:
                    self.moving_point += np.array([0, 0, 1])
                elif self.moving_point[-1] == self.upper_layer:
                    self.moving_point += np.array([0, 0, -1])

            # trajectory
            self.trajectory = np.append(self.trajectory, np.array([self.moving_point]), axis=0)
            # via_tag
            if self.prv_point[-1] != self.moving_point[-1]:
                self.via_tag.pop()
                self.via_tag.append(True)
            else:
                self.via_tag.append(False)

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            for key, value in self.grid_arrays.items():
                if int(key[-1]) in [self.lower_layer, self.upper_layer]:
                    print(f"{key} table")
                    print(f"moving metal : {self.moving_point[-1]}")
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
                                         "O": np.array([[8, 2, 2], [8, 6, 2], [2, 6, 2]])},
                 routing_grid       =   "12",
                 render_mode        =   "console"):


        super(RoutingEnv, self).__init__()
        self.metal_nodes            =   copy.deepcopy(init_metal_nodes)
        self.metal_edges            =   copy.deepcopy(init_metal_edges)

        self.lower_layer            =   int(routing_grid[0])
        self.upper_layer            =   int(routing_grid[1])

        self.render_mode            =   render_mode

        self.points_to_route        =   copy.deepcopy(points_to_route)
        
        self.max_x, self.max_y = self.metal_nodes[0][0], self.metal_nodes[0][1]
        self.min_x, self.min_y = self.metal_nodes[0][0], self.metal_nodes[0][1]
        for point in self.metal_nodes:
            if point[0] >= self.max_x: self.max_x = point[0]
            if point[1] >= self.max_y: self.max_y = point[1]
            if point[0] <= self.min_x: self.min_x = point[0]
            if point[1] <= self.min_y: self.min_y = point[1]

        #### agent generation ####
        self.agents = []
        for key, value in self.points_to_route.items():
            for i in range(len(value)-1):
                self.agents.append(Agent(name=key, metal_nodes=self.metal_nodes, metal_edges=self.metal_edges,
                                    start_point=value[i], end_point=value[i+1],
                                    max_x=self.max_x, min_x=self.min_x, max_y=self.max_y, min_y=self.min_y,
                                    upper_layer=self.upper_layer, lower_layer=self.lower_layer, render_mode = "console"))
        ##########################

        

        self.trajectories           =   [agent.trajectory for agent in self.agents]
        self.via_tags               =   [agent.via_tag for agent in self.agents]
        self.goal_reached           =   [False for _ in range(len(self.agents))]
        self.obstacle_reached       =   [False for _ in range(len(self.agents))]

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.MultiDiscrete([3 for _ in range(len(self.agents))])   # 0 : go front, 1 : go back, 2 : metal change
        obs_dict = dict()
        for i in range(len(self.agents)):
            obs_dict[f"agent_{i}"] = spaces.Box(
                low=-1, high=1,
                shape=(int(1 + 2 + 6*2**2-2*2+3),), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_dict)



    def get_rwd(self):
        reward = 0

        for agent in self.agents:
            start_goal_dist = np.linalg.norm(agent.goal_point - agent.start_point)
            agent_goal_dist = np.linalg.norm(agent.goal_point - agent.moving_point)
            prv_goal_dist = np.linalg.norm(agent.goal_point - agent.prv_point)

            reward -= (agent_goal_dist - prv_goal_dist)

            if agent.obstacle_reached:
                reward -= start_goal_dist

        return reward


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent
        for agent in self.agents:
            agent.reset()
        
        obs = self.get_obs()

        return obs

    def get_obs(self):
        obs = dict()
        for i, agent in enumerate(self.agents):
            obs[f"agent_{i}"] = agent.get_obs()
        return obs

    def check_termination(self):
        goal_reached = []
        obstacle_reached = []
        for agent in self.agents:
            agent.check_termination()
            goal_reached.append(agent.goal_reached)
            obstacle_reached.append(agent.obstacle_reached)
        self.goal_reached = goal_reached
        self.obstacle_reached = obstacle_reached


    def step(self, action):
        # action[i] == 0:   # agent i : + 1
        # action[i] == 1:   # agent i : - 1
        # action[i] == 2:   # agent i : metal change
        
        # print(self.moving_point)

        for i, agent in enumerate(self.agents):
            agent.step(action[i])

        # observation
        obs = self.get_obs()

        # termination condition
        self.check_termination()

        # REWARD
        reward = self.get_rwd()

        # done
        
        # print(np.all(self.goal_reached), self.goal_reached)
        # print(np.any(self.obstacle_reached), self.obstacle_reached)

        done = bool(np.all(self.goal_reached) or np.any(self.obstacle_reached))

        # Optionally we can pass additional info, we are not using that for now
        self.trajectories           =   [agent.trajectory for agent in self.agents]
        self.via_tags               =   [agent.via_tag for agent in self.agents]
        info = {"trajectories": self.trajectories, "via_tags": self.via_tags}

        for agent in self.agents:
            agent.prv_point = copy.deepcopy(agent.moving_point)

        return obs, reward, done, info

    def render(self):
        for agent in self.agents:
            agent.render()

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
        action = [np.random.randint(3) for _ in range(len(env.agents))]
        print(f"action : {action}")
        obs, rewards, done, info = env.step(action)
        print(obs)
        print(info)
        env.render()

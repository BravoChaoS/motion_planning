#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt

from tools import *
from rrt import *
from rrt_apf import *
from rrt_apf_sa import *
from potential_fields import *


def move_obstacles(obstacles, params):
    # obstacles[3] += np.array([0.004, 0.005])
    # small cubes movement
    obstacles[-3] += np.array([0.02, 0.0]) * params.drone_vel
    obstacles[-2] += np.array([-0.005, 0.005]) * params.drone_vel
    obstacles[-1] += np.array([0.0, 0.01]) * params.drone_vel
    return obstacles


class Params:
    def __init__(self):
        self.mode = 2
        self.animate = 0  # show RRT construction, set 0 to reduce time of the RRT algorithm
        self.visualize = 0  # show constructed paths at the end of the RRT and path smoothing algorithms
        self.maxiters = 5000  # max number of samples to build the RRT
        self.goal_prob = 0.05  # with probability goal_prob, sample the goal
        self.minDistGoal = 0.25  # [m], min distance os samples from goal to add goal node to the RRT
        self.extension = 0.1  # [m], extension parameter: this controls how far the RRT extends in each step.
        self.world_bounds_x = [-2.5, 2.5]  # [m], map size in X-direction
        self.world_bounds_y = [-2.5, 2.5]  # [m], map size in Y-direction
        self.drone_vel = 4.0  # [m/s]
        self.ViconRate = 100  # [Hz]
        self.max_sp_dist = 0.3 * self.drone_vel  # [m], maximum distance between current robot's pose and the sp from global planner
        self.influence_radius = 1.22  # potential fields radius, defining repulsive area size near the obstacle
        self.goal_tolerance = 0.05  # [m], maximum distance threshold to reach the goal
        self.num_robots = 3
        self.moving_obstacles = 0  # move small cubic obstacles or not
        self.apf_coef = 0.3  # coefficient for APF
        self.test_loops = 100
        self.file_path = 'D:\\BCSpace\\Projects\\PythonProjects\\motion_planning\\result\\'


class Robot:
    def __init__(self):
        self.sp = [0, 0]
        self.sp_global = [0, 0]
        self.route = np.array([self.sp])
        self.f = 0
        self.leader = False
        self.vel_array = []

    def local_planner(self, obstacles, params):
        obstacles_grid = grid_map(obstacles)
        self.f = combined_potential(obstacles_grid, self.sp_global, params.influence_radius)
        self.sp, self.vel = gradient_planner_next(self.sp, self.f, params)
        self.vel_array.append(norm(self.vel))
        self.route = np.vstack([self.route, self.sp])


# Initialization
params = Params()
xy_start = np.array([1.4, 1.4])
xy_goal = np.array([1.5, -0.9])
# xy_goal = np.array([1.4, 1.2])


""" Obstacles map construction """
obstacles = [
    # bugtrap
    np.array([[0.5, 0], [2.5, 0.], [2.5, 0.3], [0.5, 0.3]]),
    np.array([[0.5, 0.3], [0.8, 0.3], [0.8, 1.5], [0.5, 1.5]]),
    # np.array([[0.5, 1.5], [1.5, 1.5], [1.5, 1.8], [0.5, 1.8]]),
    # angle
    np.array([[-2, -2], [-0.5, -2], [-0.5, -1.8], [-2, -1.8]]),
    np.array([[-0.7, -1.8], [-0.5, -1.8], [-0.5, -0.8], [-0.7, -0.8]]),
    # walls
    np.array([[-2.5, -2.5], [2.5, -2.5], [2.5, -2.49], [-2.5, -2.49]]),
    np.array([[-2.5, 2.49], [2.5, 2.49], [2.5, 2.5], [-2.5, 2.5]]),
    np.array([[-2.5, -2.49], [-2.49, -2.49], [-2.49, 2.49], [-2.5, 2.49]]),
    np.array([[2.49, -2.49], [2.5, -2.49], [2.5, 2.49], [2.49, 2.49]]),

    np.array([[-1.0, 2.0], [0.5, 2.0], [0.5, 2.5], [-1.0, 2.5]]),  # my table
    np.array([[-1.0, 2.0], [0.5, 2.0], [0.5, 2.5], [-1.0, 2.5]]) + np.array([2.0, 0]),  # Evgeny's table
    np.array([[-2.0, -0.5], [-2.0, 1.0], [-2.5, 1.0], [-2.5, -0.5]]),  # Roman's table
    np.array([[-1.2, -1.2], [-1.2, -2.5], [-2.5, -2.5], [-2.5, -1.2]]),  # mats
    np.array([[2.0, 0.8], [2.0, -0.8], [2.5, -0.8], [2.5, 0.8]]),  # Mocap table

    # moving obstacle
    np.array([[-2.3, 2.0], [-2.2, 2.0], [-2.2, 2.1], [-2.3, 2.1]]),
    np.array([[2.3, -2.3], [2.4, -2.3], [2.4, -2.2], [2.3, -2.2]]),
    np.array([[0.0, -2.3], [0.1, -2.3], [0.1, -2.2], [0.0, -2.2]]),
]
passage_width = 0.25
passage_location = 0.5
barrier_right = 0.8
barrier_left = -0.8
# obstacles = [
#     # narrow passage
#     np.array(
#         [[-2.5, -0.5], [-passage_location - passage_width / 2., -0.5], [-passage_location - passage_width / 2., 0.5],
#          [-2.5, 0.5]]),
#     np.array([[-passage_location + passage_width / 2., -0.5], [2.5, -0.5],
#               [2.5, 0.5], [-passage_location + passage_width / 2., 0.5]]),
#     np.array([[-passage_location + passage_width + barrier_left, -1.5 - passage_width], [-passage_location + passage_width + barrier_right, -1.5 - passage_width],
#               [-passage_location + passage_width + barrier_right, -1.5], [-passage_location + passage_width + barrier_left, -1.5]]),
#     np.array([[-passage_location + barrier_right, -1.5], [-passage_location + passage_width + barrier_right, -1.5],
#               [-passage_location + passage_width + barrier_right, -0.5], [-passage_location + barrier_right, -0.5]]),
# ]

robots = []
for i in range(params.num_robots):
    robots.append(Robot())
robot1 = robots[0]
robot1.leader = True

# postprocessing variables:
mean_dists_array = []
max_dists_array = []

# Layered Motion Planning: RRT (global) + Potential Field (local)
if __name__ == '__main__':
    # RRT-global planning
    plt.figure(figsize=(10, 10))
    draw_map(obstacles)
    plt.plot(xy_start[0], xy_start[1], 'bo', color='red', markersize=20, label='start')
    plt.plot(xy_goal[0], xy_goal[1], 'bo', color='green', markersize=20, label='goal')

    obstacles_grid = grid_map(obstacles)

    file_path = params.file_path
    if params.mode == 0:
        file_path += 'RRT.txt'
    elif params.mode == 1:
        file_path += 'APF.txt'
    elif params.mode == 2:
        file_path += 'SA.txt'

    # two lists for storing time and number of iterations
    tms = []
    iters = []

    for i in range(params.test_loops):
        t, it = 0, 0
        if params.mode == 0:
            P_long, t, it = rrt_path(obstacles, xy_start, xy_goal, params)
        elif params.mode == 1:
            f = combined_potential(obstacles_grid, xy_goal, params.influence_radius)
            [gy, gx] = np.gradient(-f)
            draw_gradient(f)
            P_long, t, it = rrt_apf_path(obstacles, xy_start, xy_goal, params, gx, gy)
        elif params.mode == 2:
            att, rep = devided_potential(obstacles_grid, xy_goal, params.influence_radius)
            g_att = np.gradient(-att)
            g_rep = np.gradient(-rep)
            draw_gradient(att)
            draw_gradient(rep)
            P_long, t, it = rrt_apf_sa_path(obstacles, xy_start, xy_goal, params, g_att, g_rep)
        tms.append(t)
        iters.append(it)

    print('Average time: ', np.mean(tms))
    print('Average iterations: ', np.mean(iters))
    print('Average time per iteration: ', np.mean(tms) / np.mean(iters))
    # 方差
    print('Variance of time: ', np.var(tms))
    print('Variance of iterations: ', np.var(iters))
    print('Variance of time per iteration: ', np.var(tms) / np.mean(iters))

    with open(file_path, mode='a', encoding='utf-8') as f:
        f.write('Average time: ' + str(np.mean(tms)) + '\n')
        f.write('Average iterations: ' + str(np.mean(iters)) + '\n')
        f.write('Average time per iteration: ' + str(np.mean(tms) / np.mean(iters)) + '\n')
        # 方差
        f.write('Variance of time: ' + str(np.var(tms)) + '\n')
        f.write('Variance of iterations: ' + str(np.var(iters)) + '\n')
        f.write('Variance of time per iteration: ' + str(np.var(tms) / np.mean(iters)) + '\n')


    # plt.plot(P_long[:,0], P_long[:,1], linewidth=3, color='green', label='Global planner path')
    # plt.pause(1.0)
    P = ShortenPath(P_long, obstacles, smoothiters=30)  # P = [[xN, yN], ..., [x1, y1], [x0, y0]]

    traj_global = waypts2setpts(P, params)
    P = np.vstack([P, xy_start])
    plt.plot(P[:, 0], P[:, 1], linewidth=3, color='orange', label='Global planner path')
    plt.pause(0.1)

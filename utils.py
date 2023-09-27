import os
import numpy as np
import torch
from torch import nn


YAW_OFFSET_PATH = os.path.expanduser("~/kamigami_data/calibration/yaw_offsets.npy")


### TORCH UTILS ###

def to_device(*args, device=torch.device("cpu")):
    ret = []
    for arg in args:
        ret.append(arg.to(device))
    return ret if len(ret) > 1 else ret[0]

def dcn(*args):
    ret = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            ret.append(arg)
        else:
            ret.append(arg.detach().cpu().numpy())
    return ret if len(ret) > 1 else ret[0]

def as_tensor(*args):
    ret = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            ret.append(torch.as_tensor(arg, dtype=torch.float))
        else:
            ret.append(arg)
    return ret if len(ret) > 1 else ret[0]

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def create_initialize_network(input_dim, output_dim, hidden_depth, hidden_dim, layernorm=False, activation_fn=nn.ReLU):
    assert hidden_depth >= 1

    hidden_layers = []
    for _ in range(hidden_depth):
        if layernorm:
            hidden_layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), activation_fn()]
        else:
            hidden_layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]

    if layernorm:
        input_layer = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), activation_fn()]
    else:
        input_layer = [nn.Linear(input_dim, hidden_dim), activation_fn()]

    output_layer = [nn.Linear(hidden_dim, output_dim)]

    layers = input_layer + hidden_layers + output_layer
    net = nn.Sequential(*layers)
    net.apply(initialize_weights)

    return net

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def sin_cos(angle):
    return torch.sin(angle), torch.cos(angle)

def signed_angle_difference(angle1, angle2):
    return (angle1 - angle2 + 3 * torch.pi) % (2 * torch.pi) - torch.pi

def rotate_vectors(vectors, angle):
    sin, cos = sin_cos(angle)
    rotation = torch.stack((torch.stack((cos, -sin)),
                            torch.stack((sin, cos)))).permute(2, 0, 1)
    rotated_vector = (rotation @ vectors[..., None]).squeeze(dim=-1)
    return rotated_vector


class SetParamsAsAttributes:
    def __init__(self, params):
        super().__init__()

        for key, value in vars(params).items():
            self.__setattr__(key, value)

        self.params = params


class DataUtils(SetParamsAsAttributes):
    def __init__(self, params):
        super().__init__(params)

    def state_to_model_input(self, state):
        if self.n_robots == 1 and not self.use_object:
            return None

        state = as_tensor(state)

        n_states = self.n_robots + self.use_object
        relative_states = torch.empty((state.size(0), 4*(n_states-1)))

        # use object state as base state
        base_xy, base_heading = state[:, -3:-1], state[:, -1]

        for i in range(n_states-1):
            cur_state = state[:, 3*i:3*(i+1)]
            xy, heading = cur_state[:, :2], cur_state[:, 2]

            absolute_state_to_base_xy = base_xy - xy
            relative_state_to_base_xy = rotate_vectors(absolute_state_to_base_xy, -base_heading)

            relative_state_heading = signed_angle_difference(heading, base_heading)
            relative_state_sc = torch.stack(sin_cos(relative_state_heading), dim=1)

            relative_state = torch.cat((relative_state_to_base_xy, relative_state_sc), dim=1)
            relative_states[:, 4*i:4*(i+1)] = relative_state


        # base_xy, base_heading = state[:, :2], state[:, 2]

        # for i in range(n_states-1):
        #     cur_state = state[:, 3*(i+1):3*(i+2)]
        #     xy, heading = cur_state[:, :2], cur_state[:, 2]

        #     absolute_state_to_base_xy = base_xy - xy
        #     relative_state_to_base_xy = rotate_vectors(absolute_state_to_base_xy, -base_heading)

        #     relative_state_heading = signed_angle_difference(heading, base_heading)
        #     relative_state_sc = torch.stack(sin_cos(relative_state_heading), dim=1)

        #     relative_state = torch.cat((relative_state_to_base_xy, relative_state_sc), dim=1)
        #     relative_states[:, 4*i:4*(i+1)] = relative_state

        ##################

        # relative_states = torch.empty((state.size(0), 4*n_states))

        # all_idx = torch.arange(4*n_states)
        # xy_idx = all_idx.reshape((n_states, 4))[:, :2].reshape(n_states*2)
        # sin_cos_idx = all_idx.reshape((n_states, 4))[:, 2:].reshape(n_states*2)

        # state_all_idx = torch.arange(3*n_states)
        # state_xy_idx = state_all_idx.reshape((n_states, 3))[:, :2].reshape(n_states*2)
        # state_heading_idx = state_all_idx.reshape((n_states, 3))[:, 2].reshape(n_states)

        # relative_states[:, xy_idx] = state[:, state_xy_idx]
        # relative_states[:, sin_cos_idx] = torch.stack(sin_cos(state[:, state_heading_idx]), dim=-1).transpose(-2, -1).reshape(-1, n_states*2)

        return relative_states

    def state_to_model_input_tdm(self, state):
        state = as_tensor(state)
        robot_states = self.state_to_model_input(state)
        object_states = torch.zeros(*robot_states.shape[:-1], 4)
        return torch.cat([robot_states, object_states], dim=-1)

    def relative_goal_xysc(self, state, goal):
        state, goal = as_tensor(state, goal)
        object_state = state[:, -3:]

        relative_goal = torch.empty(state.size(0), 4)
        relative_goal[:, :2] = (goal - object_state)[:, :2]

        # clip xy norm between 0 and goal_clip_dist
        # xy_norm = relative_goal[:, :2].norm(dim=-1)
        # xy_unit_vector = relative_goal[:, :2] / xy_norm
        # clip_idx = torch.where(xy_norm > self.goal_clip_dist)[0]
        # relative_goal[clip_idx, :2] = xy_unit_vector[clip_idx] * self.goal_clip_dist

        heading_difference = signed_angle_difference(goal[:, -1], object_state[:, -1])
        relative_goal[:, 2:] = torch.stack(sin_cos(heading_difference), dim=1)

        return relative_goal

    def compute_relative_delta_xysc(self, state, next_state):
        state, next_state = as_tensor(state, next_state)

        n_states = self.n_robots + self.use_object
        relative_delta_xysc = torch.empty((state.size(0), 4*n_states))

        for i in range(n_states):
            cur_state, cur_next_state = state[..., 3*i:3*(i+1)], next_state[..., 3*i:3*(i+1)]

            xy, next_xy = cur_state[:, :2], cur_next_state[:, :2]
            heading, next_heading = cur_state[:, 2], cur_next_state[:, 2]

            absolute_delta_xy = next_xy - xy
            relative_delta_xy = rotate_vectors(absolute_delta_xy, -heading)

            heading_difference = signed_angle_difference(next_heading, heading)
            heading_difference_sc = torch.stack(sin_cos(heading_difference), dim=1)

            relative_delta_xysc[:, 4*i:4*(i+1)] = torch.cat((relative_delta_xy, heading_difference_sc), dim=1)

        return relative_delta_xysc

    def next_state_from_relative_delta(self, state, relative_delta):
        state, relative_delta = as_tensor(state, relative_delta)

        n_states = self.n_robots + self.use_object
        next_state = torch.empty_like(state)

        for i in range(n_states):
            cur_state, cur_relative_delta = state[:, 3*i:3*(i+1)], relative_delta[:, 4*i:4*(i+1)]

            absolute_xy, absolute_heading = cur_state[:, :2], cur_state[:, 2]
            relative_delta_xy, relative_delta_sc = cur_relative_delta[:, :2], cur_relative_delta[:, 2:]

            absolute_delta_xy = rotate_vectors(relative_delta_xy, absolute_heading)
            absolute_next_xy = absolute_xy + absolute_delta_xy

            relative_delta_heading = torch.atan2(*relative_delta_sc.T)
            absolute_next_heading = signed_angle_difference(absolute_heading, -relative_delta_heading)

            next_state[:, 3*i:3*(i+1)] = torch.cat((absolute_next_xy, absolute_next_heading[:, None]), dim=1)

        return next_state

    def cost_dict(self, state, action, goals, initial_state, robot_goals=False, signed=False):
        dist_bonus_thresh = 0.03
        dist_penalty_thresh = 0.1
        big_dist_penalty_thresh = 0.15
        sep_cost_thresh = 0.4
        realistic_cost_thresh = 0.14
        goal_object_robot_angle_thresh = 90. * torch.pi / 180.

        initial_state, action, goals = as_tensor(initial_state, action, goals)
        state_dim = 3

        robot_states = state[..., :self.n_robots*state_dim].reshape(*state.shape[:-1], self.n_robots, state_dim)
        robot_states = torch.movedim(robot_states, -2, 0)

        if self.use_object:
            object_state = state[..., -state_dim:]

            effective_state = robot_states[0] if robot_goals else object_state
        else:
            effective_state = state[..., :state_dim]

        # distance to goal position
        state_to_goal_xy = (goals - effective_state)[..., :-1]
        dist_cost = torch.norm(state_to_goal_xy, dim=-1)
        if signed:
            dist_cost *= forward

        if robot_goals:
            state_to_goal_xy_robots = (goals - robot_states)[..., :-1]
            dist_cost = torch.norm(state_to_goal_xy_robots, dim=-1).mean(dim=0)[0]

        # bonus for being very close to the goal
        dist_bonus = torch.zeros_like(dist_cost)
        dist_bonus[dist_cost.abs() < dist_bonus_thresh] = -1.
        # dist_bonus = (torch.abs(dist_cost) < 0.01) * -1

        # penalty for being far from the goal
        dist_penalty = torch.zeros_like(dist_cost)
        dist_penalty[dist_cost.abs() > dist_penalty_thresh] = 1.

        # penalty for being very far from the goal
        big_dist_penalty = torch.zeros_like(dist_cost)
        big_dist_penalty[dist_cost.abs() > big_dist_penalty_thresh] = 1.

        # difference between current and toward-goal heading
        current_angle = effective_state[..., 2]
        target_angle = torch.atan2(state_to_goal_xy[..., 1], state_to_goal_xy[..., 0])
        heading_cost = signed_angle_difference(target_angle, current_angle)

        left = (heading_cost > 0) * 2 - 1
        forward = (torch.abs(heading_cost) < torch.pi / 2) * 2 - 1

        heading_cost[forward == -1] = (heading_cost[forward == -1] + torch.pi) % (2 * torch.pi)
        heading_cost = torch.stack((heading_cost, 2 * torch.pi - heading_cost)).min(dim=0)[0]

        if signed:
            heading_cost *= left * forward
        else:
            heading_cost = torch.abs(heading_cost)

        # difference between current and specified heading
        target_angle = goals[..., 2]
        target_heading_cost = signed_angle_difference(target_angle, current_angle)

        left = (target_heading_cost > 0) * 2 - 1
        forward = (torch.abs(target_heading_cost) < torch.pi / 2) * 2 - 1

        target_heading_cost[forward == -1] = (target_heading_cost[forward == -1] + torch.pi) % (2 * torch.pi)
        target_heading_cost = torch.stack((target_heading_cost, 2 * torch.pi - target_heading_cost)).min(dim=0)[0]

        if signed:
            target_heading_cost *= left * forward
        else:
            target_heading_cost = torch.abs(target_heading_cost)

        # object state change
        if self.use_object:
            tiled_init_object_state = torch.tile(initial_state[-3:], (*object_state.shape[:-2], 1))[..., None, :]
            object_state_with_init = torch.cat([tiled_init_object_state, object_state], dim=-2)
            object_xy_delta_cost = -torch.norm(torch.diff(object_state_with_init[..., :-1], dim=-2), dim=-1)
        else:
            object_xy_delta_cost = torch.tensor([0.])

        # object-robot separation distance
        if self.use_object:
            object_to_robot_xy = (object_state - robot_states)[..., :-1]
            object_to_robot_dist = torch.norm(object_to_robot_xy, dim=-1)

            # max_object_to_robot_dist = object_to_robot_dist.max(dim=0)[0]
            # sep_cost = torch.zeros_like(max_object_to_robot_dist)
            # sep_cost[max_object_to_robot_dist > sep_cost_thresh] = 1.

            # sep_cost = object_to_robot_dist.mean(dim=0)
            # sep_cost[max_object_to_robot_dist < sep_cost_thresh] = 0.

            sep_cost = object_to_robot_dist.clone()
            sep_cost[object_to_robot_dist < sep_cost_thresh] = 0.
            sep_cost = sep_cost.mean(dim=0)
        else:
            sep_cost = torch.tensor([0.])

        # realistic distance
        # separation distance between each robot and the distance between object and robots
        if self.use_object and self.n_robots > 1:
            object_to_robot_dist = torch.norm((object_state - robot_states)[..., :-1], dim=-1)
            robot_to_robot_dist = torch.norm((robot_states[0] - robot_states[1])[..., :-1], dim=-1)
            all_dists = torch.cat((object_to_robot_dist, robot_to_robot_dist[None, ...]), dim=0)

            # all_dists = torch.clip(all_dists, 0., realistic_cost_thresh)
            # realistic_cost = (realistic_cost_thresh - all_dists).mean(dim=0)

            min_dists = all_dists.min(dim=0)[0]
            realistic_cost = torch.zeros_like(min_dists, dtype=torch.float)
            realistic_cost[min_dists < realistic_cost_thresh] = 1.
        else:
            realistic_cost = torch.tensor([0.])

        # object-robot heading difference
        if self.use_object:
            robot_theta, object_theta = robot_states[..., -1], object_state[..., -1]
            heading_diff = (robot_theta - object_theta) % (2 * torch.pi)
            heading_diff_cost = torch.stack((heading_diff, 2 * torch.pi - heading_diff), dim=1).min(dim=1)[0]
        else:
            heading_diff_cost = torch.tensor([0.])

        # action magnitude
        norm_cost = -torch.norm(action, dim=-1)[:, None, :]

        # to-object heading
        if self.use_object:
            robot_xy = robot_states[..., :2]
            robot_heading = robot_states[..., -1]

            object_xy = object_state[..., :2]
            state_to_object_xy = (object_xy - robot_xy)

            target_heading = torch.atan2(state_to_object_xy[..., 1], state_to_object_xy[..., 0])
            to_object_heading_cost = signed_angle_difference(target_heading, robot_heading)

            left = (to_object_heading_cost > 0) * 2 - 1
            forward = (torch.abs(to_object_heading_cost) < torch.pi / 2) * 2 - 1

            to_object_heading_cost[forward == -1] = (to_object_heading_cost[forward == -1] + torch.pi) % (2 * torch.pi)
            to_object_heading_cost = torch.stack((to_object_heading_cost, 2 * torch.pi - to_object_heading_cost)).min(dim=0)[0]

            if signed:
                to_object_heading_cost *= left * forward
            else:
                to_object_heading_cost = torch.abs(to_object_heading_cost)
                to_object_heading_cost = to_object_heading_cost.mean(dim=0)
        else:
            to_object_heading_cost = torch.tensor([0.])

        # goal-object-robot angle
        if self.use_object:
            robot_xy = robot_states[..., :2]

            object_xy = object_state[..., :2]
            goal_xy = goals[..., :2]

            a = torch.norm(object_xy - robot_xy, dim=-1)
            b = torch.norm(goal_xy - object_xy, dim=-1)
            c = torch.norm(goal_xy - robot_xy, dim=-1)

            goal_object_robot_angle_cost_per_robot = torch.pi - torch.arccos(((a**2 + b**2 - c**2) / (2 * a * b + 1e-6)).clamp(-1., 1.))
            goal_object_robot_angle_cost = goal_object_robot_angle_cost_per_robot.mean(dim=0) / torch.pi

            # goal_object_robot_angle_cost[dist_cost < 0.03] = 0.
            # goal_object_robot_angle_cost[goal_object_robot_angle_cost < torch.pi / 6.] = 0.

            # max_goal_object_robot_angle_cost = goal_object_robot_angle_cost_per_robot.max(dim=0)[0]
            # goal_object_robot_angle_cost = torch.zeros_like(max_goal_object_robot_angle_cost, dtype=torch.float)
            # goal_object_robot_angle_cost[max_goal_object_robot_angle_cost > goal_object_robot_angle_thresh] = 1.
            # goal_object_robot_angle_cost[max_goal_object_robot_angle_cost < goal_object_robot_angle_thresh] = 0.

            # goal_object_robot_angle_cost = goal_object_robot_angle_cost_per_robot.clone()
            # goal_object_robot_angle_cost[goal_object_robot_angle_cost_per_robot < goal_object_robot_angle_thresh] = 0.
            # goal_object_robot_angle_cost = goal_object_robot_angle_cost.mean(dim=0)
        else:
            goal_object_robot_angle_cost = torch.tensor([0.])

        # print("\n\n", dist_cost[:, :, 0].mean(), "\n")

        # print("\n\n", dist_bonus[:, :, 0].mean(), "|", dist_penalty[:, :, 0].mean(), "\n")

        # print("\n\n", sep_cost[:, :, 0].mean(), "\n")

        # print(f'\n\ngoal_obj_robot_angle 0: {goal_object_robot_angle_cost_per_robot[0, :, :, 0].mean() * 180. / torch.pi}')
        # print(f'goal_obj_robot_angle 2: {goal_object_robot_angle_cost_per_robot[1, :, :, 0].mean() * 180. / torch.pi}\n')

        # import time
        # time.sleep(2.)

        # dist_cost = dist_cost ** 2

        dist_cost -= dist_cost.min()
        dist_cost /= dist_cost.max()

        cost_dict = {
            "distance": dist_cost,                                      # distance to goal
            "distance_bonus": dist_bonus,                               # bonus for being close to goal
            "distance_penalty": dist_penalty,                           # penalty for being far from goal
            "big_distance_penalty": big_dist_penalty,                   # bonus for being very close to goal
            "heading": heading_cost,                                    # difference between goal and current heading
            "target_heading": target_heading_cost,                      # difference between current heading and heading toward goal
            "separation": sep_cost,                                     # penalizing robot separation from object
            "heading_difference": heading_diff_cost,                    # difference between object and robot heading
            "action_norm": norm_cost,                                   # norm of the action
            "to_object_heading": to_object_heading_cost,                # difference between current heading and heading toward object
            "goal_object_robot_angle": goal_object_robot_angle_cost,    # penalty for deviating from 180deg
            "realistic": realistic_cost,                                # penalizes unrealistic predictions based on interference
            "object_delta": object_xy_delta_cost,                       # encourages large object xy deltas
        }

        return cost_dict

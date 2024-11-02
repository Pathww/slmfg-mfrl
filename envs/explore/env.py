import numpy as np
import copy
from envs.utils import *

class Explore2d(object):
    def __init__(self, args):
        self.env_name = 'explore'
        self.args = args
        self.agent_num = args.agent_num
        self.map = [
            '###########',
            '#         #',
            '#         #',
            '#         #',
            '#         #',
            '#         #',
            '#         #',
            '#         #',
            '#         #',
            '#         #',
            '###########',
        ]
        self.action_to_moveX = [0, 0, -1, 1, 0]
        self.action_to_moveY = [1, -1, 0, 0, 0]
        self.map_M, self.map_N = 11, 11
        self.node_num = self.map_M * self.map_N
        self.node_xy = [ids_1dto2d(i, self.map_M, self.map_N) for i in range(self.node_num)]

        self.action_size = 5
        self.obs_dim = 2 * self.map_M + args.episode_len + 1    # +1 to allow for t == horizon [0, ..., horizon]
        self.init_node_id = np.array([1 + self.map_M] * self.agent_num)
        self.cur_node_id = copy.deepcopy(self.init_node_id)
        self.pre_node_id = copy.deepcopy(self.init_node_id)
        self.cur_node_xy = np.array([self.node_xy[self.cur_node_id[agent]] for agent in range(self.agent_num)])

        self.node_action_node_map = np.zeros((self.node_num, self.action_size))
        for node_id in range(self.node_num):
            node_x, node_y = ids_1dto2d(node_id, self.map_M, self.map_N)
            if node_y + 1 < self.map_N and self.map[node_x][node_y + 1] != '#':
                self.node_action_node_map[node_id, 0] = ids_2dto1d(node_x, node_y + 1, self.map_M, self.map_N)
            else:
                self.node_action_node_map[node_id, 0] = node_id
            if node_y - 1 >= 0 and self.map[node_x][node_y - 1] != '#':
                self.node_action_node_map[node_id, 1] = ids_2dto1d(node_x, node_y - 1, self.map_M, self.map_N)
            else:
                self.node_action_node_map[node_id, 1] = node_id
            if node_x - 1 >= 0 and self.map[node_x - 1][node_y] != '#':
                self.node_action_node_map[node_id, 2] = ids_2dto1d(node_x - 1, node_y, self.map_M, self.map_N)
            else:
                self.node_action_node_map[node_id, 2] = node_id
            if node_x + 1 < self.map_M and self.map[node_x + 1][node_y] != '#':
                self.node_action_node_map[node_id, 3] = ids_2dto1d(node_x + 1, node_y, self.map_M, self.map_N)
            else:
                self.node_action_node_map[node_id, 3] = node_id
            self.node_action_node_map[node_id, 4] = node_id

    def reset(self):
        self.cur_node_id = copy.deepcopy(self.init_node_id)
        self.pre_node_id = copy.deepcopy(self.init_node_id)
        self.cur_node_xy = np.array([self.node_xy[self.cur_node_id[agent]] for agent in range(self.agent_num)])
        return self.get_obs(0)

    def reset_agent_pos(self, agent_num=None):
        if agent_num is None:
            pass
        self.agent_num = agent_num
        self.init_node_id = np.array([1 + self.map_M] * self.agent_num)
        self.cur_node_id = copy.deepcopy(self.init_node_id)
        self.pre_node_id = copy.deepcopy(self.init_node_id)

    def get_agent_dist(self, normalize=True):
        agent_dist = np.array([len(np.where(self.cur_node_id == node_id)[0]) for node_id in range(self.node_num)])
        if normalize and np.sum(agent_dist) > 0:
            agent_dist = agent_dist / self.agent_num
        return agent_dist

    def get_obs(self, t):
        cur_t = np.zeros((self.agent_num, self.args.episode_len + 1))
        cur_t[:, t] = 1.
        node_id = np.zeros((self.agent_num, self.map_M + self.map_N))
        for agent in range(self.agent_num):
            node_id[agent, self.node_xy[self.cur_node_id[agent]][0]] = 1.
            node_id[agent, self.map_M + self.node_xy[self.cur_node_id[agent]][1]] = 1.
        feasible_action = np.ones((self.agent_num, self.action_size))
        obs = np.concatenate((node_id, cur_t), 1)

        return obs.astype(np.float32, copy=False), feasible_action.astype(np.float32, copy=False)

    def step(self, t, actions, act_mask, agent_type='homo'):
        agent_rewards = np.zeros(self.agent_num)

        # We only compute the representative agent's reward as only this agent's
        # experience tuple (s, a, r, s') will be used to learn the policy
        agent = 0
        agent_dist = self.get_agent_dist()
        agent_rewards[agent] += (-self.args.aversion_coe * np.log(agent_dist[self.cur_node_id[agent]] + 1e-20))

        # However, we need to complete the transitions of all the agents as the distribution
        # will be used to compute the representative agent's reward
        node_action_node = self.node_action_node_map[self.pre_node_id, :]
        self.pre_node_id = copy.deepcopy(self.cur_node_id)
        self.cur_node_id = node_action_node[range(node_action_node.shape[0]), actions].astype(np.int32, copy=False)

        return agent_rewards
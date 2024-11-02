import numpy as np
import copy

class Crowd(object):
    def __init__(self, args):
        self.env_name = 'crowd'
        self.args = args
        self.agent_num = args.agent_num
        self.positional_reward_xy = [5, 15]
        self.action_to_moveX = [-1, 0, 1]
        self.map_M = 20
        self.node_num = self.map_M

        self.action_size = 3
        self.obs_dim = self.map_M + args.episode_len + 1    # +1 to allow for t == horizon [0, ..., horizon]
        self.init_node_id = np.array([0] * self.agent_num)
        self.cur_node_id = copy.deepcopy(self.init_node_id)
        self.pre_node_id = copy.deepcopy(self.init_node_id)

        # Position on the circle [0, map_M)
        self.node_action_node_map = np.zeros((self.node_num, self.action_size))
        for node_id in range(self.node_num):
            if node_id - 1 >= 0:
                self.node_action_node_map[node_id, 0] = node_id - 1
            else:
                self.node_action_node_map[node_id, 0] = self.node_num - 1
            if node_id + 1 < self.map_M:
                self.node_action_node_map[node_id, 2] = node_id + 1
            else:
                self.node_action_node_map[node_id, 2] = 0
            self.node_action_node_map[node_id, 1] = node_id

    def reset(self):
        self.cur_node_id = copy.deepcopy(self.init_node_id)
        self.pre_node_id = copy.deepcopy(self.init_node_id)
        return self.get_obs(0)

    def reset_agent_pos(self, agent_num=None):
        if agent_num is None:
            pass
        self.agent_num = agent_num
        self.init_node_id = np.array([0] * self.agent_num)
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
        node_id = np.eye(self.node_num)[self.cur_node_id]
        obs = np.concatenate((node_id, cur_t), 1)
        feasible_action = np.ones((self.agent_num, self.action_size))
        return obs.astype(np.float32, copy=False), feasible_action.astype(np.float32, copy=False)

    def step(self, t, actions, act_mask, agent_type='homo'):
        agent_rewards = np.zeros(self.agent_num)

        # We only compute the representative agent's reward as only this agent's
        # experience tuple (s, a, r, s') will be used to learn the policy
        agent = 0
        agent_dist = self.get_agent_dist()
        agent_rewards[agent] += (-self.args.aversion_coe * np.log(agent_dist[self.cur_node_id[agent]] + 1e-20))
        if t < self.args.episode_len / 2:
            if self.cur_node_id[agent] == self.positional_reward_xy[0]:
                agent_rewards[agent] += 5.
        else:
            if self.cur_node_id[agent] == self.positional_reward_xy[1]:
                agent_rewards[agent] += 5.

        # However, we need to complete the transitions of all the agents as the
        # distribution will be used to compute the representative agent's reward
        node_action_node = self.node_action_node_map[self.pre_node_id, :]
        self.pre_node_id = copy.deepcopy(self.cur_node_id)
        self.cur_node_id = node_action_node[range(node_action_node.shape[0]), actions].astype(np.int32, copy=False)
        return agent_rewards
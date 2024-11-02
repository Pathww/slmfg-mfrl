import numpy as np

from .objects import *

class TaxiSimulator(object):
    def __init__(self, args):
        self.env_name = 'taxi'
        self.args = args
        self.agent_num = args.agent_num

        #objects map and manager
        self.map = Map(args) 
        self.agent_manager = Agent_Manager(args, self.map.node_num, self.map.max_speed)
        
        self.map.update_agent_info(self.agent_manager.cur_node_id)
        self.map_M, self.map_N = args.map_M, args.map_N

        # size of action set
        self.action_size = self.agent_manager.action_size
        # size of obs: x + y + t
        self.obs_dim = args.map_M + args.map_N + args.episode_len + 1 if args.map_str == 'grid' else 1 + args.episode_len + 1

    def reset(self):
        self.agent_manager.reset()
        self.map.update_agent_info(self.agent_manager.cur_node_id)

    def reset_agent_pos(self, agent_num=None):
        if agent_num is None:
            pass
        np.random.seed(self.args.seed)
        self.agent_num = agent_num
        self.map.total_agent_num = agent_num
        np.random.seed(self.args.seed)
        self.agent_manager.init_node_id = np.random.randint(self.map.node_num, size=agent_num).astype(np.int32)
        self.agent_manager.reset()
        self.map.update_agent_info(self.agent_manager.cur_node_id)

    def get_agent_dist(self, normalize=True):
        agent_dist = self.map.get_agent_num()
        if normalize and np.sum(agent_dist) > 0:
            agent_dist /= self.agent_num
        return agent_dist

    def get_obs(self, t):
        cur_t = np.zeros((self.agent_num, self.args.episode_len + 1))
        cur_t[:, t] = 1.
        if self.args.map_str == 'grid':
            node_id = np.zeros((self.agent_num, self.args.map_M + self.args.map_N))
            for agent in range(self.agent_num):
                node_id[agent, self.map.nodes[self.agent_manager.cur_node_id[agent]].xy[0]] = 1.
                node_id[agent, self.args.map_M + self.map.nodes[self.agent_manager.cur_node_id[agent]].xy[1]] = 1.
            feasible_action = np.ones((self.agent_num, self.action_size))
        else:
            node_id = self.agent_manager.cur_node_id / self.map.node_num
            node_id = node_id[:, np.newaxis]
            feasible_action = np.array([
                self.map.feasible_action[self.agent_manager.cur_node_id[i]]
                [min(self.agent_manager.agent_speed, self.args.speed)]
                for i in range(self.agent_num)])
        obs = np.concatenate((node_id, cur_t), 1)

        return obs.astype(np.float32, copy=False), feasible_action.astype(np.float32, copy=False)

    def step(self, t, actions, act_mask, representative_agent=True):
        if self.args.map_str == 'grid':
            cur_agent_node_id, new_agent_node_id = self.agent_manager.set_new_node_id_by_direction(actions)
            agent_final_node_id, agent_rewards = self.map.step(t, cur_agent_node_id, new_agent_node_id, representative_agent)
        else:
            if self.args.act_mask and self.args.speed < self.map.max_speed:
                mask = act_mask[range(act_mask.shape[0]), actions]
                new_agent_node_id = np.where(mask == 1, actions, self.agent_manager.cur_node_id)
            else:
                new_agent_node_id = actions
            agent_final_node_id, agent_rewards = self.map.step(t, self.agent_manager.cur_node_id, new_agent_node_id)
            self.agent_manager.set_new_node_id(new_agent_node_id)
        return agent_rewards
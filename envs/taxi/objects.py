import numpy as np
import copy
from envs.utils import *

class Node(object):
    def __init__(self, node_id, map_M, map_N, episode_len, seed, distance):
        self.map_M = map_M
        self.map_N = map_N
        self.xy = ids_1dto2d(node_id, map_M, map_N)
        self.node_id = node_id  # unique node id, start from 0
        self.episode_len = episode_len
        self.seed = seed
        self.orders = [[] for _ in range(episode_len)]   # order list at each time step
        self.order_num = np.zeros((episode_len,), dtype=np.int32)  # order num at each time step
        self.distance = distance

        self.agent_num = 0

    def set_orders(self, orders):
        self.orders = orders
        for t in range(self.episode_len):
            self.order_num[t] = len(orders[t])

    def update_agent_info(self, agent_node_id):
        self.agent_num = len(np.where(agent_node_id == self.node_id)[0])

    def compute_reward(self, t, total_agent_num, aversion=False):
        agent_final_node_id = np.array([self.node_id] * self.agent_num)
        agent_rewards = np.zeros(self.agent_num)
        if self.order_num[t] > 0 and self.agent_num > 0:
            avg_price = -sum([order['price'] for order in self.orders[t]]) * np.log(self.agent_num / total_agent_num + 1e-20)
            agent_rewards = avg_price * np.ones(self.agent_num)
        if self.agent_num > 1 and aversion:
            agent_rewards -= np.log((self.agent_num - 1) / total_agent_num)

        return agent_final_node_id, agent_rewards


class Map(object):
    """
    Consists of all the nodes
    """
    def __init__(self, args):
        self.args = args
        np.random.seed(args.seed)

        # determine the feasible actions according to adjacent matrix and feasible speed
        if args.map_str == 'manhattan':  # manhattan map
            # original_node_id: taxi zone ids from TLC NYC dataset <==> node_id: 0, 1, ...
            original_node_id, node_id = [], []
            with open('envs/taxi/data/manhattan/manhattan_node_ids.txt') as f:
                for line in f.readlines():
                    temp = line.split()
                    original_node_id.append(int(temp[0]))
                    node_id.append(int(temp[1]))
            original_node_id = np.array(original_node_id)
            node_id = np.array(node_id)
            self.node_num = len(node_id)
            graph_matrix = np.zeros((self.node_num, self.node_num))
            adj_matrix = np.zeros((self.node_num, self.node_num))
            self.distance = np.zeros((self.node_num, self.node_num))
            path_cost = []
            with open('envs/taxi/data/manhattan/manhattan_distance.txt') as f:
                for line in f.readlines():
                    temp = line.split()
                    begin_node_id = node_id[np.where(original_node_id == int(temp[0]))[0][0]]
                    end_node_id = node_id[np.where(original_node_id == int(temp[1]))[0][0]]
                    items = [float(begin_node_id), float(end_node_id), float(temp[2])]
                    path_cost.append(items)
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if i == j:
                        graph_matrix[i][j] = 0
                    else:
                        graph_matrix[i][j] = 999
            i = 0
            while i < len(path_cost):
                start_point = int(path_cost[i][0])
                end_point = int(path_cost[i][1])
                adj_matrix[start_point][end_point] = 1
                adj_matrix[end_point][start_point] = 1
                graph_matrix[start_point][end_point] = 1
                graph_matrix[end_point][start_point] = 1
                self.distance[start_point][end_point] = 1
                self.distance[end_point][start_point] = 1
                i += 1
        elif args.map_str == 'grid':  # grid map
            self.node_num = args.map_M * args.map_N
            graph_matrix = np.zeros((self.node_num, self.node_num))
            adj_matrix = np.zeros((self.node_num, self.node_num))
            self.distance = np.zeros((self.node_num, self.node_num))
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if i == j:
                        graph_matrix[i][j] = 0
                    else:
                        graph_matrix[i][j] = 999
            for i in range(args.map_M):
                for j in range(args.map_N):
                    cur_node_id = ids_2dto1d(i, j, args.map_M, args.map_N)
                    if i - 1 >= 0:
                        adj_matrix[cur_node_id][ids_2dto1d(i - 1, j, args.map_M, args.map_N)] = 1
                        adj_matrix[ids_2dto1d(i - 1, j, args.map_M, args.map_N)][cur_node_id] = 1
                        graph_matrix[cur_node_id][ids_2dto1d(i - 1, j, args.map_M, args.map_N)] = 1
                        graph_matrix[ids_2dto1d(i - 1, j, args.map_M, args.map_N)][cur_node_id] = 1
                        self.distance[cur_node_id][ids_2dto1d(i - 1, j, args.map_M, args.map_N)] = 1
                        self.distance[ids_2dto1d(i - 1, j, args.map_M, args.map_N)][cur_node_id] = 1
                    if i + 1 < args.map_M:
                        adj_matrix[cur_node_id][ids_2dto1d(i + 1, j, args.map_M, args.map_N)] = 1
                        adj_matrix[ids_2dto1d(i + 1, j, args.map_M, args.map_N)][cur_node_id] = 1
                        graph_matrix[cur_node_id][ids_2dto1d(i + 1, j, args.map_M, args.map_N)] = 1
                        graph_matrix[ids_2dto1d(i + 1, j, args.map_M, args.map_N)][cur_node_id] = 1
                        self.distance[cur_node_id][ids_2dto1d(i + 1, j, args.map_M, args.map_N)] = 1
                        self.distance[ids_2dto1d(i + 1, j, args.map_M, args.map_N)][cur_node_id] = 1
                    if j - 1 >= 0:
                        adj_matrix[cur_node_id][ids_2dto1d(i, j - 1, args.map_M, args.map_N)] = 1
                        adj_matrix[ids_2dto1d(i, j - 1, args.map_M, args.map_N)][cur_node_id] = 1
                        graph_matrix[cur_node_id][ids_2dto1d(i, j - 1, args.map_M, args.map_N)] = 1
                        graph_matrix[ids_2dto1d(i, j - 1, args.map_M, args.map_N)][cur_node_id] = 1
                        self.distance[cur_node_id][ids_2dto1d(i, j - 1, args.map_M, args.map_N)] = 1
                        self.distance[ids_2dto1d(i, j - 1, args.map_M, args.map_N)][cur_node_id] = 1
                    if j + 1 < args.map_N:
                        adj_matrix[cur_node_id][ids_2dto1d(i, j + 1, args.map_M, args.map_N)] = 1
                        adj_matrix[ids_2dto1d(i, j + 1, args.map_M, args.map_N)][cur_node_id] = 1
                        graph_matrix[cur_node_id][ids_2dto1d(i, j + 1, args.map_M, args.map_N)] = 1
                        graph_matrix[ids_2dto1d(i, j + 1, args.map_M, args.map_N)][cur_node_id] = 1
                        self.distance[cur_node_id][ids_2dto1d(i, j + 1, args.map_M, args.map_N)] = 1
                        self.distance[ids_2dto1d(i, j + 1, args.map_M, args.map_N)][cur_node_id] = 1
        else:
            raise ValueError(f"Choose map from manhattan or grid, no {args.map_str}")
        # Floyd method to get the shortest distance between any two nodes
        for i in range(self.node_num):
            for j in range(self.node_num):
                self.distance[i][j] = graph_matrix[i][j]
        for k in range(self.node_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if self.distance[i][k] + self.distance[k][j] < self.distance[i][j]:
                        self.distance[i][j] = self.distance[i][k] + self.distance[k][j]  # distance from i to i (itself) is 0
        # feasible neighbor nodes at different speed levels for each node
        self.max_speed = int(np.max(np.max(self.distance, axis=1)))
        self.feasible_action = {i: np.zeros((self.max_speed, self.node_num)) for i in range(self.node_num)}
        for i in range(self.node_num):
            for speed in range(self.max_speed):
                self.feasible_action[i][speed][np.where(self.distance[i] <= speed + 1)[0]] = 1
        self.normalized_distance = self.distance / self.max_speed

        self.total_agent_num = args.agent_num
        self.nodes = [Node(i, args.map_M, args.map_N, args.episode_len + 1, args.seed,
                           self.normalized_distance[i]) for i in range(self.node_num)]

        # generate the order information
        order = [[[] for _ in range(args.episode_len + 1)] for _ in range(self.node_num)]
        if args.order_dist == 'uniform':
            for t in range(args.episode_len + 1):
                for i in range(args.order_num):
                    node_id = np.random.randint(self.node_num)
                    price = np.random.choice(list(range(args.order_price_min, args.order_price_max)))
                    order[node_id][t].append({'begin_node': node_id, 'end_node': np.random.randint(self.node_num), 'price': price})
        elif args.order_dist == 'gaussian':
            for t in range(args.episode_len + 1):
                np.random.seed(args.seed + t)
                for i in range(args.order_num):
                    x = int(np.clip(np.random.normal(int(args.map_M / 2), 2), 0, args.map_M - 1))
                    y = int(np.clip(np.random.normal(int(args.map_N / 2), 2), 0, args.map_N - 1))
                    node_id = ids_2dto1d(x, y, args.map_M, args.map_N)
                    price = np.random.choice(list(range(args.order_price_min, args.order_price_max)))
                    order[node_id][t].append({'begin_node': node_id, 'end_node': np.random.randint(self.node_num), 'price': price})
        else:
            raise ValueError(f"Choose order distribution from uniform or gaussian, no {args.order_dist}")
        for i in range(self.node_num):
            self.nodes[i].set_orders(order[i])
        self.order_dist = [np.array([self.nodes[i].order_num[t] for i in range(self.node_num)], dtype=np.float32)
                                   for t in range(args.episode_len + 1)]
        for t in range(args.episode_len + 1):
            if np.sum(self.order_dist[t]) > 0:
                self.order_dist[t] /= np.sum(self.order_dist[t])

    def get_agent_num(self):
        return np.array([self.nodes[i].agent_num for i in range(self.node_num)]).astype(np.float32, copy=False)

    def get_order_num(self, t):
        return np.array([self.nodes[i].order_num[t] for i in range(self.node_num)])

    def update_agent_info(self, agent_node_id):
        for i in range(self.node_num):
            self.nodes[i].update_agent_info(agent_node_id)

    def step(self, t, cur_agent_node_id, new_agent_node_id, representative_agent=False):
        agent_final_node_id = copy.deepcopy(new_agent_node_id)
        agent_rewards = np.zeros(self.total_agent_num)
        if representative_agent:
            agent = 0
            for i in range(self.node_num):
                agent_ids = np.where(new_agent_node_id == i)[0]  # ndarray
                self.nodes[i].agent_num = len(agent_ids)
            agent_ids = np.where(new_agent_node_id == new_agent_node_id[agent])[0]
            final_node_id, reward = self.nodes[new_agent_node_id[agent]].\
                compute_reward(t, self.total_agent_num, self.args.aversion)
            agent_final_node_id[agent_ids] = final_node_id
            agent_rewards[agent_ids] = reward
        else:
            for i in range(self.node_num):
                agent_ids = np.where(new_agent_node_id == i)[0]  # ndarray
                self.nodes[i].agent_num = len(agent_ids)
                if len(agent_ids) > 0:
                    final_node_id, reward = self.nodes[i].compute_reward(t, self.total_agent_num, self.args.aversion)
                    if self.args.cost_coe > 0:
                        distance_agent_ids = self.normalized_distance[cur_agent_node_id[agent_ids],:]
                        distance = distance_agent_ids[range(distance_agent_ids.shape[0]), new_agent_node_id[agent_ids]]
                        reward -= self.args.cost_coe * distance
                    if self.args.penalty_move:
                        reward -= np.array(cur_agent_node_id[agent_ids] != new_agent_node_id[agent_ids]).astype(float) * len(agent_ids) / self.total_agent_num
                    agent_final_node_id[agent_ids] = final_node_id
                    agent_rewards[agent_ids] = reward

        return agent_final_node_id, agent_rewards



class Agent_Manager(object):
    def __init__(self, args, node_num, max_speed):
        np.random.seed(args.seed)
        self.map_M = args.map_M
        self.map_N = args.map_N
        self.args = args
        self.node_num = node_num
        if args.map_str == 'grid':
            self.action_size = 5    # up, down, left, right, stay
        else:
            self.action_size = self.node_num    # action num. = num. of nodes
        self.init_node_id = np.random.randint(self.node_num, size=args.agent_num).astype(np.int32)
        self.cur_node_id = copy.deepcopy(self.init_node_id)
        self.pre_node_id = copy.deepcopy(self.init_node_id)
        self.agent_speed = min(args.speed, max_speed) - 1

        self.node_action_node_map = np.zeros((self.node_num, self.action_size))
        for node_id in range(self.node_num):
            node_x, node_y = ids_1dto2d(node_id, self.map_M, self.map_N)
            # up
            if node_y + 1 < self.map_N:
                self.node_action_node_map[node_id, 0] = ids_2dto1d(node_x, node_y + 1, self.map_M, self.map_N)
            else:
                self.node_action_node_map[node_id, 0] = node_id
            # down
            if node_y - 1 >= 0:
                self.node_action_node_map[node_id, 1] = ids_2dto1d(node_x, node_y - 1, self.map_M, self.map_N)
            else:
                self.node_action_node_map[node_id, 1] = node_id
            # left
            if node_x - 1 >= 0:
                self.node_action_node_map[node_id, 2] = ids_2dto1d(node_x - 1, node_y, self.map_M, self.map_N)
            else:
                self.node_action_node_map[node_id, 2] = node_id
            # right
            if node_x + 1 < self.map_M:
                self.node_action_node_map[node_id, 3] = ids_2dto1d(node_x + 1, node_y, self.map_M, self.map_N)
            else:
                self.node_action_node_map[node_id, 3] = node_id
            # stay
            self.node_action_node_map[node_id, 4] = node_id

    def reset(self):
        self.cur_node_id = copy.deepcopy(self.init_node_id)
        self.pre_node_id = copy.deepcopy(self.init_node_id)

    def set_new_node_id_by_direction(self, action):
        node_action_node = self.node_action_node_map[self.pre_node_id, :]
        self.pre_node_id = copy.deepcopy(self.cur_node_id)
        self.cur_node_id = node_action_node[range(node_action_node.shape[0]), action].astype(np.int32, copy=False)
        return self.pre_node_id, self.cur_node_id

    def set_new_node_id(self, new_node_id):
        self.pre_node_id = copy.deepcopy(self.cur_node_id)
        self.cur_node_id = new_node_id
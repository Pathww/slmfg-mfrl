import random
import numpy as np
import torch

from envs.utils import ids_1dto2d  

class CriticalAgents:
    def __init__(self, adv_num, agent_num):
        self.adv_num = adv_num
        self.agent_num = agent_num

    def random_agents(self):
        random_list = random.sample(range(self.agent_num), self.adv_num)
        adv_agents = sorted(random_list)
        return adv_agents

    def center_agents(self, init_node_id, width): #  2x2 4x4 6x6
        adv_agents = []
        xm = 4.5
        ym = 4.5
        width = width // 2
        x, y = ids_1dto2d(init_node_id)
        for i in range(self.agent_num):
            if (abs(x[i]-xm)<=width) and (abs(y[i]-ym)<=width):
                adv_agents.append(i)
        return adv_agents
    
    def edge_agents(self, init_node_id, width): # 4*2 4*4 4*6
        adv_agents = []
        xm = 4.5
        ym = 4.5
        width = width // 2
        x, y = ids_1dto2d(init_node_id)
        for i in range(self.agent_num):
            if (x[i] == 0 and (abs(y[i]-ym)<=width)) or (x[i] == 9 and (abs(y[i]-ym)<=width)) or ((abs(x[i]-xm)<=width) and y[i] == 0) or ((abs(x[i]-xm)<=width) and y[i] == 9):
                adv_agents.append(i)
        return adv_agents
    
    def corner_agents(self, init_node_id, width): # 4x1^1 4x2^2 4x3^3
        adv_agents = []
        width = width - 1
        x, y = ids_1dto2d(init_node_id)
        for i in range(self.agent_num):
            if ((abs(x[i]-0)<=width) and (abs(y[i]-0)<=width)) or ((abs(x[i]-9)<=width) and (abs(y[i]-0)<=width)) or ((abs(x[i]-0)<=width) and (abs(y[i]-9)<=width)) or ((abs(x[i]-9)<=width) and (abs(y[i]-9)<=width)):
                adv_agents.append(i)
        return adv_agents
    
    def dc_agents(self, init_node_id):
        degrees =  np.zeros(self.agent_num)
        x, y = ids_1dto2d(init_node_id)
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                if i == j:
                    continue
                if (x[j] == x[i] and y[j] == y[i]) or (x[j] == x[i]+1 and y[j] == y[i]) or (x[j] == x[i]-1 and y[j] == y[i]) or (x[j] == x[i] and y[j] == y[i]+1) or (x[j] == x[i] and y[j] == y[i]-1):
                    degrees[i] = degrees[i] + 1

        nodes_sorted_by_degree = np.argsort(-degrees)  # 使用负号实现降序排列

        top_nodes = []
       
        i = 0
        while i < len(nodes_sorted_by_degree):
            current_degree = degrees[nodes_sorted_by_degree[i]]
            same_degree_nodes = [node for node in nodes_sorted_by_degree[i:] if degrees[node] == current_degree]
            
            if len(top_nodes) + len(same_degree_nodes) > self.adv_num:
                remaining = self.adv_num - len(top_nodes)
                top_nodes.extend(random.sample(same_degree_nodes, remaining))  # 从相同度数的节点中随机选择剩余的节点
                break
            else:
                top_nodes.extend(same_degree_nodes)  # 否则全部加入
                i += len(same_degree_nodes)  # 更新索引，跳过这些相同度数的节点
        result = sorted(top_nodes)
        return result
    
    def ours_agents(self, vnet, obs, init_node_id):
        self.vnet = vnet
        eps = 0

        agent_num = self.agent_num
        xis = np.zeros(agent_num)
        mask = np.zeros(agent_num)
        adv_agents = []

        V_ = self.vnet.V(obs, eps, xis)

        for i in range(self.adv_num):
            temp_xis = xis.copy()
            max_reward = -1e9
            target = -1
            for k in range(agent_num):
                if mask[k] == 0:
                    temp_xis_k = temp_xis.copy()
                    temp_xis_k[k] = 1
                    temp_eps = (i+1)/agent_num
                    
                    V = self.vnet.V(obs, temp_eps, temp_xis_k)
                    reward = torch.sum(V_ - V)/agent_num
                
                    if reward > max_reward:
                        xis = temp_xis_k.copy()  
                        V_max = V.clone()    
                        target = k
                        max_reward = reward.clone()  
                    
            V_ = V_max.clone()
            mask[target] = 1
            adv_agents.append(target)
        return adv_agents


import random

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

        
        


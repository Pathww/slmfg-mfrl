from envs.explore.env import Explore2d
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--agent-num', type=int, default=100, help='Number of agents')
parser.add_argument('--episode-len', type=int, default=20, help='Number of agents')
parser.add_argument('--aversion-coe', type=int, default=1, help='Number of agents')

if __name__ == "__main__":
    args = parser.parse_args()
    env = Explore2d(args)
    obs, act_mask = env.reset()
    t = 0
    while t < args.episode_len:
        actions = np.random.randint(0, env.action_size, args.agent_num)
        rewards = env.step(t, actions, act_mask)
        t += 1
        print('t ', t, 'act ', actions[0], ' ', rewards[0])
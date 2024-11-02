import torch.nn as nn
import torch
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def visualize(N_list, dist_list, T_list, env_name):
    plt.figure(figsize=(24, 6))
    sns.set()
    sns.set(style="whitegrid")
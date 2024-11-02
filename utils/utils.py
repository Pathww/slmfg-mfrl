import torch.nn as nn
import torch
import numpy as np

def update_epsilon(init_epsilon, finial_epsilon, cur_step, min_step, max_step):
    if max_step == 0:
        return 0.0
    elif cur_step <= min_step:
        return init_epsilon
    elif cur_step > max_step:
        return finial_epsilon
    else:
        inter = (init_epsilon - finial_epsilon) / (max_step - min_step)
        return init_epsilon - inter * (cur_step - min_step)

def weights_init_uniform(m):
    """initializing weights"""
    a = -1
    b = 1
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.uniform_(a, b)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(a, b)
        m.bias.data.fill_(0)
    elif classname.find('Embedding') != -1:
        m.weight.data.uniform_(a, b)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

def init_weights_kaiming_uniform(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

def mean_weights(m):
    if type(m) == nn.Linear:
        print('w ', torch.mean(m.weight), ' b ', torch.mean(m.bias))

def save_model(model, path):
    """save trained model parameters"""
    torch.save(model.state_dict(), path)

def load_model(model, path, avoid=None):
    """load trained model parameters"""
    state_dict = dict(torch.load(path))
    if avoid is not None:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith(avoid)}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(state_dict)

def to_Cuda(tensor, cuda, cuda_id=None):
    """convert a tensor to a cuda tensor"""
    if cuda_id is None:
        return tensor.cuda() if cuda else tensor
    else:
        return tensor.cuda(cuda_id) if cuda else tensor

def sample_unseen_task(min_N, max_N, tasks=[], num_of_unseen=1, method='uniform', lam=1):
    sampled = [False] * (max_N - min_N)
    unseen_task = []
    for i in range(num_of_unseen):
        if method == 'uniform':
            while True:
                a_num = np.random.randint(min_N, max_N - 1)
                if not sampled[a_num] and a_num not in tasks:
                    break
        elif method == 'poisson':
            while True:
                a_num = np.random.poisson(lam, size=1)
                if min_N <= a_num <= max_N - 1 and not sampled[a_num] and a_num not in tasks:
                    break
        elif method == 'exponent':
            while True:
                a_num = int(np.round(np.random.exponential(lam, size=1)))
                if min_N <= a_num <= max_N - 1 and not sampled[a_num] and a_num not in tasks:
                    break
        else:
            raise ValueError(f"Unknown env sample method: {method}")
        unseen_task.append(a_num)
        sampled[a_num] = True

    return unseen_task

def l1_reg_loss(model):
    reg_loss = None
    for param in model.parameters():
        if reg_loss is None:
            reg_loss = torch.sum(torch.abs(param))
        else:
            reg_loss += torch.sum(torch.abs(param))
    return reg_loss

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, max_encode=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / max_encode ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sin_pos_embed_from_grid(embed_dim, pos, max_encode=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim, dtype=np.float32)
    omega /= embed_dim
    omega = 1.0 / max_encode ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    # print(out)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb_cos
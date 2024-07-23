from jax import random
import jax.numpy as jnp
import numpy as np
import scipy.io as sio
from scipy.integrate import odeint
import pandas as pd

def generate_data(T = -1, T_ext = 3):
    key = random.PRNGKey(321)
    keys = random.split(key, num = 3)
    cort_nodes_df = pd.read_csv('tau_data/cortical_regions.txt', header = None)
    L = np.load('tau_data/L.npy')
    cort_nodes = cort_nodes_df.values-1
    cort_nodes = cort_nodes[:,0]
    num_regions, num_samples = 83, 76
    
    t = np.linspace(0, T*np.ones(num_samples), T+1).T
    t_dense = np.linspace(0, t[:,-1], 100).T
    t_extra = np.linspace(0, np.ones(num_samples)*T_ext, 100).T
    c0 = np.load('tau_data/c0.npy')
    mean_c0, std_c0 = np.mean(c0, axis = 0, keepdims = True), np.std(c0, axis = 0, keepdims = True)
    lower_c0, upper_c0 = -mean_c0/std_c0, (1-mean_c0)/std_c0
    c0 = random.truncated_normal(keys[0], lower_c0, upper_c0, c0.shape) * std_c0 + mean_c0
    num_groups = 4
    indices = np.arange(num_samples) % 4
    alpha_noise, alpha_meta_noise, kappa_noise = 0.2, 0.1, 0.5
    alpha_mean, kappa_mean = 0.6, 1
    kappa_lower, kappa_upper = -kappa_mean/kappa_noise, float('inf')
    alpha_meta = alpha_mean + alpha_meta_noise * random.normal(keys[0], (num_groups,))[indices]
    alpha = alpha_meta + alpha_noise * random.normal(keys[1], (num_samples,))
    kappa = kappa_mean + kappa_noise * random.truncated_normal(keys[2], kappa_lower, kappa_upper, (num_samples,))
    c = np.array(list(map(lambda i: odeint(h, c0[i], t[i], args=(kappa[i], alpha[i], indices[i], L)), range(num_samples))))
    c_dense = np.array(list(map(lambda i: odeint(h, c0[i], t_dense[i], args=(kappa[i], alpha[i], indices[i], L)), range(num_samples))))
    c_extra = np.array(list(map(lambda i: odeint(h, c0[i], t_extra[i], args=(kappa[i], alpha[i], indices[i], L)), range(num_samples))))
    t, t_dense, t_extra = t[..., None], t_dense[..., None], t_extra[..., None]
    return t, t_dense, t_extra, c, c_dense, c_extra, cort_nodes, num_samples, num_regions, num_groups, L, alpha, kappa, indices

def reaction(c, alpha, index):
    if index == 0:
        return alpha * c * (1 - c)
    elif index == 1:
        return alpha * c * (1 - c ** 2) * 3 ** 1.5 / 8
    elif index == 2:
        return alpha * c * (1 - c ** 3) * 2 ** (2/3) / 3
    elif index == 3:
        return alpha * c * (1 - c) * np.exp(c - 1) / 4 / (5**0.5-2) / (np.exp(0.5*(5**0.5-3)))
    else:
        raise NotImplementedError
        
def h(c, t, kappa, alpha, index, L):
    return - kappa * L @ c + reaction(c, alpha, index)
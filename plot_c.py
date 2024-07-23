#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:07:45 2023

@author: zzhang99
"""

import matplotlib.pyplot as plt
from net import multi_fnn_index, multi_fnn_index_fixed_bd_norm
from data import generate_data
import numpy as np
from scipy.integrate import odeint
from pysr import PySRRegressor
from jax import jit
import jax.numpy as jnp
from utils import parse_region_index, get_mean_std, get_min_max, reverse
import argparse
import pickle

@jit
def h_nn(c, t, i, f_params, kappa, alpha):
    cc = jnp.tile(c, [num_samples, 1, 1]).reshape([num_samples, -1, 1])
    f = fnn_f(f_params, cc, indices)[i].squeeze()
    return - kappa * L @ c + alpha.squeeze() * f

def h_symbolic(c, t, i, model, kappa, alpha):
    f = model.predict(c[:, None])
    return - kappa * L @ c + alpha.squeeze() * f

def plot_c_sim(path, region_name):
    region_index = parse_region_index(region_name)
    colors = ['r', 'b']
    kappa = np.exp(kappa_params)
    c_nn_extra = np.array(list(map(lambda i: odeint(h_nn, c_data[i, 0], t_extra[i].squeeze(), args=(i, f_params, kappa[i], alpha_params[i])), range(selected_samples))))
    
    plt.figure(figsize=(36,18))
    for i in range(selected_samples):
        row, col = i % 4, i // 4
        plt.subplot(4,9,row*9+col+1)
        model = PySRRegressor.from_file(f"models/{path}_{i%4}.pkl")
        c_symbolic_extra = odeint(h_symbolic, c_data[i, 0], t_extra[i].squeeze(), args=(i, model, kappa[i], alpha_params[i]))
        for j, color in zip(region_index, colors):
            plt.plot(t[i,:], c_data[i,:,j], f'{color}x', markersize = 12, label = 'Training data')  
            plt.plot(t_extra[i,:], c_symbolic_extra[:,j], f'{color}--', alpha = 0.6, dashes = (3,3), linewidth = 5, label = 'Extrapolation with symbolic model')
            plt.plot(t_extra[i,:], c_data_extra[i,:,j], f'{color}', alpha = 0.6, label = 'Ground truth')
            plt.plot(t_extra[i,::10], c_nn_extra[i,::10,j], f'{color}v', mfc = 'none', markersize = 12, label = 'Extrapolation with PINN')
        if col == 0 and row == 3:
            plt.ylabel('tau', fontsize = 26)
            plt.xlabel('Time [yrs]', fontsize = 26)
        if col == 0:
            plt.yticks([0,1], fontsize = 26)
        else:
            plt.yticks([])
        if row == 3:
            plt.xticks([0,20], fontsize = 26)
        else:
            plt.xticks([])
        plt.xlim(0,20)
        plt.ylim(0,1.3)
        plt.title(fr'$\alpha = {(alpha_params[i]).item():4.2f}$, $\kappa = {kappa[i].item():4.2f}$', fontsize = 26)
    plt.tight_layout()
    plt.savefig(f'figs/{path}_{region_name}_c_sim.pdf')
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='symbolic')
    parser.add_argument('--region', default='entorhinal', type=str, help='region name')
    args = parser.parse_args()
    region_name = args.region
    T = 2
    selected_samples = 36
    t, t_dense, t_extra, c_data, c_data_dense, c_data_extra, mask, num_samples, num_regions, num_groups, L, alpha_data, kappa_data, indices = generate_data(T, 30)
    fixed_bd = True
    if fixed_bd:
        fnn_f = multi_fnn_index_fixed_bd_norm
    else:
        fnn_f = multi_fnn_index
    reverse_indices = reverse(indices)

    seed = 123
    path = f'{T}_{seed}'
    alpha_d = alpha_data.reshape([-1, num_groups])
    c_params, alpha_params, f_params, kappa_params = pickle.load(open(f'saved/params_{path}', 'rb'))
    plot_c_sim(path, region_name)
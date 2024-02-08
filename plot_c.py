#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:07:45 2023

@author: zzhang99
"""

import matplotlib.pyplot as plt
from net import multi_fnn_index, multi_fnn, multi_fnn_fixed_bd
import os
from example_jax import generate_data
import numpy as np
from scipy.integrate import odeint
from pysr import PySRRegressor
from jax import jit
import jax.numpy as jnp

@jit
def f(c, t, i, f_params, kappa, alpha):
    cc = jnp.tile(c, [num_samples, 1, 1]).reshape([num_samples, -1, 1])
    f = fnn_f(f_params, cc, indices)[i].squeeze()
    return - kappa * L @ c + alpha.squeeze() * f

def f_symbolic(c, t, i, model, kappa, alpha):
    if data_type == 'sim':
        f = model.predict(c[:, None]) / scale[i % 4]
    else:
        f = model.predict(c[:, None])
    return - kappa * L @ c + alpha.squeeze() * f

def parse_region_index(region_name):
    with open('nodes_regionName.txt') as f:
        name_list = f.readlines()
    region_names = np.array([region_name in name for name in name_list])
    return region_names.nonzero()[0]

def plot_c(path, region_name):
    selected_samples = 36
    region_index = parse_region_index(region_name)
    colors = ['r', 'b']
    kappa = np.exp(kappa_params)
    c_pred_extra = np.array(list(map(lambda i: odeint(f, c_data[i, 0], t_extra[i].squeeze(), args=(i, f_params, kappa[i], alpha_params[i])), range(selected_samples))))
    
    plt.figure(figsize=(36,18))
    for i in range(selected_samples):
        row, col = i % 4, i // 4
        plt.subplot(4,9,row*9+col+1)
        model = PySRRegressor.from_file(f"models/{path}_{i%4}.pkl")
        c_symbolic_extra = odeint(f_symbolic, c_data[i, 0], t_extra[i].squeeze(), args=(i, model, kappa[i], alpha_params[i]))
        for j, color in zip(region_index, colors):
            plt.plot(t[i,:], c_data[i,:,j], f'{color}x', markersize = 12, label = 'Training data')  
            plt.plot(t_extra[i,:], c_symbolic_extra[:,j], f'{color}--', alpha = 0.6, dashes = (3,3), linewidth = 5, label = 'Extrapolation with symbolic model')
            plt.plot(t_extra[i,:], c_data_extra[i,:,j], f'{color}', alpha = 0.6, label = 'Ground truth')
            plt.plot(t_extra[i,::10], c_pred_extra[i,::10,j], f'{color}v', mfc = 'none', markersize = 12, label = 'Extrapolation with PINN')
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
        plt.title(fr'$\alpha = {(alpha_params[i]/scale[i % 4]).item():4.2f}$, $\kappa = {kappa[i].item():4.2f}$', fontsize = 26)
    plt.tight_layout()
    plt.savefig(f'figs/{path}_{region_name}_c.pdf')
    plt.show()
    plt.close()

def get_min_max(x):
    return np.min(x, axis=0), np.max(x, axis=0)

def get_mean_std(x):
    return np.mean(x, axis=0), np.std(x, axis=0)
    
def plot_c_real(path, region_name):
    selected_samples = 36
    region_index = parse_region_index(region_name)
    labels = {0: r'$A\beta-$', 1: r'$A\beta+$'}
    alpha_mean, alpha_std = get_mean_std(alpha_params)
    kappa_mean, kappa_std = get_mean_std(kappa_params)
    positive_indices = np.where(alpha_mean>0)[0][:selected_samples]
    
    colors = ['r', 'b']
    plt.figure(figsize=(36,18))
    for k, i in enumerate(positive_indices):
        print(k)
        row, col = k % 4, k // 4
        ax = plt.subplot(4,9,row*9+col+1)
        c_symbolic = []
        for seed in seeds:
            path = f'{data_type}_{T}_{seed}'
            model = PySRRegressor.from_file(f"models/{path}_{indices[i]}.pkl")
            c_symbolic.append(odeint(f_symbolic, c_data[i, 0], t_extra[i].squeeze(), args=(i, model, kappa_params[seed-1][i], alpha_params[seed-1][i])))
        c_symbolic = np.array(c_symbolic)
        c_symbolic_min, c_symbolic_max = get_min_max(c_symbolic)
        c_symbolic_mean, _ = get_mean_std(c_symbolic)
        for j, color in zip(region_index, colors):
            plt.plot(t[i,:], c_data[i,:,j], f'{color}x', markersize = 12, label = 'Training data')  
            plt.plot(t_extra[i,:], c_symbolic_mean[:,j], f'{color}--', alpha = 0.6, dashes = (3,3), linewidth = 5, label = 'Extrapolation with symbolic model')
            plt.fill_between(t_extra[i,:], c_symbolic_min[:,j], c_symbolic_max[:,j], color = color, alpha = 0.2)

        if col == 0 and row == 3:
            plt.ylabel('tau', fontsize = 26)
            plt.xlabel('Time [yrs]', fontsize = 26)
        if col == 0:
            plt.yticks([0,1], fontsize = 26)
        else:
            plt.yticks([])
        if row == 3:
            plt.xticks([0,30], fontsize = 26)
        else:
            plt.xticks([])
        plt.xlim(0,30)
        plt.ylim(0,1.3)
        plt.text(.01, .99, labels[indices[i]], ha='left', va='top', transform=ax.transAxes, fontsize = 26)
    plt.tight_layout()
    plt.savefig(f'figs/{path}_{region_name}_c_real.pdf')
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    os.makedirs('figs', exist_ok = True)
    data_type = 'real'
    region_name = 'entorhinal'
    #region_name = 'middletemporal'
    #region_name = 'superiortemporal'
    T = 2
    t, t_dense, t_extra, c_data, c_data_dense, c_data_extra, mask, num_samples, num_regions, num_groups, L, alpha_data, kappa_data, indices = generate_data(data_type, T, 30)
    fnn_f = multi_fnn_fixed_bd
    reverse_indices = {}
    for i in range(num_samples):
        reverse_indices[indices[i]] = reverse_indices.get(indices[i], []) + [i]
    reverse_indices = {key: np.array(value) for key, value in reverse_indices.items()}

    if data_type == 'sim':
        path = f'{data_type}_{T}'
        alpha_d = alpha_data.reshape([-1, num_groups])
        c_params, alpha_params, f_params, kappa_params = np.load(f'saved/params_{path}.npy', allow_pickle = True)
        alpha = alpha_params.reshape([-1, num_groups])
        scale = np.mean(alpha, axis = 0)/np.mean(alpha_d, axis = 0)
        plot_c(path, region_name)
    else:
        seeds = np.arange(1, 11)
        c_params, alpha_params, f_params, kappa_params = [], [], [], []
        for seed in seeds:
            path = f'{data_type}_{T}_{seed}'
            c_param, alpha_param, f_param, kappa_param = np.load(f'saved/params_{path}.npy', allow_pickle = True)
            c_params.append(c_param)
            for index in [0, 1]:
                scale = np.loadtxt(f'expressions/scale_{path}_{index}.txt')
                alpha_param = alpha_param.at[reverse_indices[index]].set(alpha_param[reverse_indices[index]]/scale)
            alpha_params.append(alpha_param)
            f_params.append(f_param)
            kappa_params.append(kappa_param)
        alpha_params = np.array(alpha_params).squeeze()
        kappa_params = np.exp(np.array(kappa_params).squeeze())
        t_extra = t_extra.squeeze()
        plot_c_real(path, region_name)
        

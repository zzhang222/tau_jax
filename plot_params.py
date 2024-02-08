#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:29:08 2023

@author: zzhang99
"""
from net import multi_fnn_index, multi_fnn_fixed_bd
from example_jax import generate_data, reaction
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import os
from scipy.stats import gaussian_kde
from pysr import PySRRegressor

def plot_params_sim():
    fig = plt.figure(figsize = (12,4))
    plt.figtext(0.83, 0.92, '$f(c)$', fontsize=12)
    
    X = np.linspace(min(np.exp(kappa_params.squeeze()))-0.8, max(np.exp(kappa_params.squeeze()))+0.8, 101)
    kappa_pred = gaussian_kde(np.exp(kappa_params.squeeze()), 0.5)
    kappa_pred._compute_covariance()
    kappa_pred_density = kappa_pred(X)
    ax = plt.subplot2grid((2,6), (0,0), colspan = 2, rowspan = 2)
    kappa_true = gaussian_kde(kappa_data.squeeze(), 0.5)
    kappa_true._compute_covariance()
    kappa_true_density = kappa_true(X)
    plt.plot(X, kappa_true_density, 'g--', label = 'Ground truth', dashes = (5,5))
    plt.fill_between(X, np.zeros_like(X), kappa_pred_density, color='g', label = 'PINN', alpha = 0.3)
    plt.title(r'Distribution of $\kappa$')
    plt.legend()
    
    colors = ['r', 'g', 'b', 'y']
    reverse_indices = {}
    for i in range(num_samples):
        reverse_indices[indices[i]] = reverse_indices.get(indices[i], []) + [i]
    reverse_indices = {key: np.array(value) for key, value in reverse_indices.items()}
    alpha_scales = [1] * num_groups
    plt.subplot2grid((2,6), (0,2), colspan = 2, rowspan = 2)
    labels = []
    for i in range(num_groups):
        alpha = alpha_params.squeeze()[reverse_indices[i]]
        X = np.linspace(0, 1.6, 101)
        alpha_d = alpha_data.squeeze()[reverse_indices[i]]
        scale = np.mean(alpha)/np.mean(alpha_d)
        alpha_scales[i] = scale
        alpha_true = gaussian_kde(alpha_d, 0.5)
        alpha_true._compute_covariance()
        alpha_true_density = alpha_true(X)
        p1, = plt.plot(X, alpha_true_density, colors[i]+'--', dashes = (5,5))
        alpha_pred = gaussian_kde(alpha / alpha_scales[i], 0.5)
        alpha_pred._compute_covariance()
        alpha_pred_density = alpha_pred(X)
        p2 = plt.fill_between(X, np.zeros_like(X), alpha_pred_density, color = colors[i], alpha = 0.3)
        labels.append((p1,p2))
    plt.legend(labels, [f'Type {i+1}' for i in range(num_groups)],
                   handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.title(r'Distribution of $\alpha$')
    
    start_pos = [(0,4), (0,5), (1,4), (1,5)]
    c = np.linspace(0,1,100)
    cc = np.tile(c[None,:,None], (num_samples,1,1))
    f = fnn_f(f_params, cc, indices)
    for i in range(num_groups):
        model = PySRRegressor.from_file(f"models/{path}_{i}.pkl")
        f_symbolic = model.predict(c[:, None])
        plt.subplot2grid((2,6), start_pos[i])
        c_index, f_index = cc[reverse_indices[i]].flatten(), f[reverse_indices[i]].flatten()
        idx = np.argsort(c_index)
        c_index, f_index = c_index[idx], f_index[idx]
        plt.plot(c_index, f_index * alpha_scales[i], 'b', alpha = 0.6, label = 'PINN')
        plt.plot(c, f_symbolic, 'g', alpha = 0.6, label = 'symbolic')
        true_f = reaction(c_index, 1, i%4)
        plt.plot(c_index, true_f, 'k--', dashes = (5,5), label = 'Ground truth')
        plt.xticks([0,1], fontsize = 8)
        plt.yticks(fontsize = 8, rotation = 90)
        plt.ylabel(f'Type {i+1}')
        if i == 0:
            plt.legend(fontsize = 'x-small')
    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(f'figs/params_{path}.pdf')
    
def plot_params_real():
    fig = plt.figure(figsize = (10,4))
    plt.figtext(0.9, 0.92, '$f(c)$', fontsize=12)
    
    X = np.linspace(-0.6, 1.6, 101)
    kappa_pred = gaussian_kde(kappa_params.flatten(), 0.5)
    kappa_pred._compute_covariance()
    kappa_pred_density = kappa_pred(X)
    ax = plt.subplot2grid((2,5), (0,0), colspan = 2, rowspan = 2)
    plt.fill_between(X, np.zeros_like(X), kappa_pred_density, color='g', label = 'PINN', alpha = 0.3)
    plt.title(r'Distribution of $\kappa$')
    plt.legend()
    
    colors = ['r', 'g']
    labels = [r'$A\beta-$', r'$A\beta+$']
    plt.subplot2grid((2,5), (0,2), colspan = 2, rowspan = 2)
    for i in range(num_groups):
        alpha = alpha_params.squeeze()[reverse_indices[i]]
        X = np.linspace(-3, 3, 101)
        alpha_pred = gaussian_kde(alpha.flatten(), 0.5)
        alpha_pred._compute_covariance()
        alpha_pred_density = alpha_pred(X)
        plt.fill_between(X, np.zeros_like(X), alpha_pred_density, color = colors[i], label = labels[i], alpha = 0.3)
    plt.legend()
    plt.title(r'Distribution of $\alpha$')
    
    start_pos = [(0,4), (1,4)]
    c = np.linspace(0,1,100)
    cc = np.tile(c[None,:,None], (num_samples,1,1))
    score_negative, score_positive = [], []
    for i in range(num_groups):
        plt.subplot2grid((2,5), start_pos[i])
        alpha = alpha_params.squeeze()[reverse_indices[i]]
        for j, seed in enumerate(seeds):
            path = f'{data_type}_{T}_{seed}'
            scale = np.loadtxt(f'expressions/scale_{path}_{i}.txt')
            f = fnn_f(f_params[j], cc, indices) * scale
            model = PySRRegressor.from_file(f"models/{path}_{i}.pkl")
            eqs = model.equations_
            if i == 0:
                score_negative.append(np.array(eqs.score).max())
            else:
                score_positive.append(np.array(eqs.score).max())
            f_symbolic = model.predict(c[:, None])
            c_index, f_index = cc[reverse_indices[i]].flatten(), f[reverse_indices[i]].flatten()
            idx = np.argsort(c_index)
            c_index, f_index = c_index[idx], f_index[idx]
            if j == 0:
                plt.plot(c_index, f_index, 'b', alpha = 0.6, label = 'PINN')
                plt.plot(c, f_symbolic, 'g', alpha = 0.6, label = 'symbolic')
            else:
                plt.plot(c_index, f_index, 'b', alpha = 0.6)
                plt.plot(c, f_symbolic, 'g', alpha = 0.6)
            plt.yticks(fontsize = 8, rotation = 90)
        if i == 0:
            plt.legend()
        plt.ylabel(labels[i])
    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    np.savetxt(f'expressions/score_negative.txt', np.array(score_negative))
    np.savetxt(f'expressions/score_positive.txt', np.array(score_positive))
    plt.savefig(f'figs/params_{path}.pdf')
    
if __name__ == '__main__':
    os.makedirs('figs', exist_ok = True)
    data_type = 'real'
    T = 2
    seeds = np.arange(1,11)
    t, t_dense, t_extra, c_data, c_data_dense, c_data_extra, mask, num_samples, num_regions, num_groups, L, alpha_data, kappa_data, indices = generate_data(data_type, T)
    reverse_indices = {}
    for i in range(num_samples):
        reverse_indices[indices[i]] = reverse_indices.get(indices[i], []) + [i]
    reverse_indices = {key: np.array(value) for key, value in reverse_indices.items()}

    if data_type == 'sim':
        fnn_f = multi_fnn_fixed_bd
        plot_params_sim()
    else:
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
        kappa_params = kappa_params[0]
        alpha_params = alpha_params[0]
        fnn_f = multi_fnn_fixed_bd
        plot_params_real()
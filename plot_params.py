#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:29:08 2023

@author: zzhang99
"""
from net import multi_fnn_index, multi_fnn_index_fixed_bd_norm
from data import generate_data, reaction
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
from scipy.stats import gaussian_kde
from pysr import PySRRegressor
from utils import reverse
import argparse
import pickle

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
    plt.subplot2grid((2,6), (0,2), colspan = 2, rowspan = 2)
    labels = []
    for i in range(num_groups):
        alpha = alpha_params.squeeze()[reverse_indices[i]]
        X = np.linspace(0, 1.2, 101)
        alpha_d = alpha_data.squeeze()[reverse_indices[i]]
        alpha_true = gaussian_kde(alpha_d, 0.5)
        alpha_true._compute_covariance()
        alpha_true_density = alpha_true(X)
        p1, = plt.plot(X, alpha_true_density, colors[i]+'--', dashes = (5,5))
        alpha_pred = gaussian_kde(alpha, 0.5)
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
    for i in range(num_groups):
        model = PySRRegressor.from_file(f"models/{path}_{i}.pkl")
        f = fnn_f(f_params, cc, indices) #* scale
        f_symbolic = model.predict(c[:, None])
        plt.subplot2grid((2,6), start_pos[i])
        c_index, f_index = cc[reverse_indices[i]].flatten(), f[reverse_indices[i]].flatten()
        idx = np.argsort(c_index)
        c_index, f_index = c_index[idx], f_index[idx]
        plt.plot(c_index, f_index, 'b', alpha = 0.6, label = 'PINN')
        plt.plot(c, f_symbolic, 'g', alpha = 0.6, label = 'symbolic')
        true_f = reaction(c_index, 1, i%4)
        plt.plot(c_index, true_f, 'k--', dashes = (5,5), label = 'Ground truth')
        plt.xticks([0,1], fontsize = 8)
        plt.yticks(fontsize = 8, rotation = 90)
        plt.ylabel(f'Type {i+1}')
        if i == 0:
            plt.legend(fontsize = 'x-small')
    _, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(f'figs/params_{path}.pdf')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='symbolic')
    args = parser.parse_args()
    T = 2
    t, t_dense, t_extra, c_data, c_data_dense, c_data_extra, mask, num_samples, num_regions, num_groups, L, alpha_data, kappa_data, indices = generate_data(T)
    reverse_indices = reverse(indices)
    fixed_bd = True
    if fixed_bd:
        fnn_f = multi_fnn_index_fixed_bd_norm
    else:
        fnn_f = multi_fnn_index
    seed = 123
    path = f'{T}_{seed}'
    _, alpha_params, f_params, kappa_params = pickle.load(open(f'saved/params_{path}', 'rb'))
    plot_params_sim()
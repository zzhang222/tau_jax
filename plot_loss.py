#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:19:09 2023

@author: zzhang99
"""

import matplotlib.pyplot as plt
import numpy as np
from pysr import PySRRegressor

def plot_loss_sim():
    print_every_adam = 10
    num_group = 4
    T = 2
    data_type = 'sim'
    path = f'{data_type}_{T}'
    loss_adam = np.load(f'saved/loss_adam_{path}.npy')
    iter_adam = np.arange(len(loss_adam)) * print_every_adam
    loss_lbfgs = np.load(f'saved/loss_lbfgs_{path}.npy')
    lbfgs_coeff = 1 / len(loss_lbfgs) * len(loss_adam) * print_every_adam
    iter_lbfgs = iter_adam[-1] + np.arange(1, len(loss_lbfgs) + 1) * lbfgs_coeff
    start_pos = [(0,2), (0,3), (1,2), (1,3)]
    
    plt.figure(figsize = (12, 4))
    plt.figtext(0.63, 0.94, 'Symbolic regression loss & score', fontsize=12)
    ax1 = plt.subplot2grid((2,4), (0,0), colspan = 2, rowspan = 2)
    ax1.set_ylim([1e-8, 1e-2])
    ax1.set_xlabel('Iterations')
    ax1.set_xticks([0, 40000, 80000, iter_adam[-1] + 10 * lbfgs_coeff, iter_adam[-1] + 20 * lbfgs_coeff], [0, 40000, 80000, 10, 20])
    ax1.set_yscale('log')
    
    axes = []
    for i in range(num_group):
        ax = plt.subplot2grid((2,4), start_pos[i], colspan = 1, rowspan = 1)
        ax.set_ylim(1e-10, 100)
        ax.set_xlim(3, 12)
        ax.set_yscale('log')
        ax.set_xticks([5, 10])
        ax.set_yticks([1e-10, 1e-6, 1e-2, 1e2])
        ax.tick_params(axis='both', which='major', labelsize=5)
        axes.append(ax)
    axes[2].set_xlabel('Complexity')
    
    
    ax1.plot(iter_adam, loss_adam, alpha = 0.6, color = 'r', linewidth = 1, label = 'Adam')
    ax1.plot([iter_adam[-1], *iter_lbfgs], [loss_adam[-1], *loss_lbfgs], alpha = 0.6, color = 'b', label = 'L-BFGS')
    ax1.vlines(100000, 1e-8, 1e-2, linestyle='dashed', color = 'k')
    
    for index in range(num_group):
        print(f"models/{path}_{index}.pkl")
        model = PySRRegressor.from_file(f"models/{path}_{index}.pkl")
        eqs = model.equations_
        score = np.array(eqs.score)
        loss = np.array(eqs.loss)
        complexity = np.array(eqs.complexity)
        if index == 2:
            label_loss = 'loss'
            label_score = 'score'
        else:
            label_loss = None
            label_score = None
        axes[index].plot(complexity, loss, linestyle = '-', color = 'r', alpha = 0.6, label = label_loss)
        axes[index].plot(complexity, score, linestyle = '--', color = 'b', alpha = 0.6, label = label_score)
        axes[index].set_ylabel(f'Type {index+1}', fontsize = 10)
    
    axes[2].legend()
    ax1.legend()
    ax1.set_title('PINN loss')
    plt.tight_layout()
    plt.savefig('figs/statistics_sim.pdf')

def plot_loss_sim_appendix():
    print_every_adam = 10
    num_group = 4
    T_list = [2, 4, 6]
    color_list = ['r', 'g', 'b']
    start_pos = [(0,2), (0,3), (1,2), (1,3)]
    
    plt.figure(figsize = (10, 4))
    plt.figtext(0.63, 0.94, 'Symbolic regression loss & score', fontsize=12)
    ax1 = plt.subplot2grid((2,4), (0,0), colspan = 2, rowspan = 2)
    ax1.set_ylim([1e-8, 1e-2])
    ax1.set_xlabel('Iterations')
    ax1.set_xticks([0, 50000, 100000])
    ax1.set_yscale('log')
    
    axes = []
    for i in range(num_group):
        ax = plt.subplot2grid((2,4), start_pos[i], colspan = 1, rowspan = 1)
        ax.set_ylim(1e-10, 100)
        ax.set_yscale('log')
        ax.set_xticks([5, 10])
        ax.set_yticks([1e-10, 1e-6, 1e-2, 1e2])
        ax.tick_params(axis='both', which='major', labelsize=5)
        axes.append(ax)
    axes[2].set_xlabel('Complexity')
    
    for color, T in zip(color_list, T_list):
        data_type = 'sim'
        path = f'{data_type}_{T}'
        loss_adam = np.load(f'saved/loss_adam_{path}.npy')
        iter_adam = np.arange(len(loss_adam)) * print_every_adam
        
        ax1.plot(iter_adam, loss_adam, alpha = 0.6, color = color, label = f'$T = {T}$')
        
        for index in range(num_group):
            print(f"models/{path}_{index}.pkl")
            model = PySRRegressor.from_file(f"models/{path}_{index}.pkl")
            eqs = model.equations_
            score = np.array(eqs.score)
            loss = np.array(eqs.loss)
            complexity = np.array(eqs.complexity)
            if index == 2:
                label_loss = f'$T = {T}$, loss'
                label_score = f'$T = {T}$, score'
            else:
                label_loss = None
                label_score = None
            axes[index].plot(complexity, loss, linestyle = '-', color = color, alpha = 0.6, label = label_loss)
            axes[index].plot(complexity, score, linestyle = '--', color = color, alpha = 0.6, label = label_score)
            axes[index].set_ylabel(f'Type {index+1}', fontsize = 7)
    
    axes[2].legend(fontsize = 4, ncols = 3)
    ax1.legend()
    ax1.set_title('PINN loss')
    plt.tight_layout()
    plt.savefig('figs/statistics_sim_appendix.pdf')
    
def plot_loss_real_appendix():
    print_every_adam = 10
    num_group = 2
    start_pos = [(0,2), (1,2)]
    
    plt.figure(figsize = (8, 4))
    plt.figtext(0.6, 0.93, 'Symbolic regression loss & score', fontsize=12)
    ax1 = plt.subplot2grid((2,4), (0,0), colspan = 2, rowspan = 2)
    ax1.set_ylim([1e-4, 1e-1])
    ax1.set_xlabel('Iterations')
    ax1.set_xticks([0, 50000, 100000])
    ax1.set_yscale('log')
    
    axes = []
    for i in range(num_group):
        ax = plt.subplot2grid((2,4), start_pos[i], colspan = 2, rowspan = 1)
        ax.set_ylim(1e-4, 100)
        ax.set_yscale('log')
        ax.set_xticks([5, 10])
        ax.set_yticks([1e-4, 1e-2, 1, 1e2])
        ax.tick_params(axis='both', which='major', labelsize=5)
        axes.append(ax)
    axes[1].set_xlabel('Complexity')
    
    data_type = 'real'
    T = 2
    path = f'{data_type}_{T}'
    loss_adam = np.load(f'saved/loss_adam_{path}.npy')
    iter_adam = np.arange(len(loss_adam)) * print_every_adam
    
    ax1.plot(iter_adam, loss_adam, alpha = 0.6, color = 'r')
    
    for index in range(num_group):
        print(f"models/{path}_{index}.pkl")
        model = PySRRegressor.from_file(f"models/{path}_{index}.pkl")
        eqs = model.equations_
        score = np.array(eqs.score)
        loss = np.array(eqs.loss)
        complexity = np.array(eqs.complexity)
        if index == 1:
            label_loss = 'loss'
            label_score = 'score'
        else:
            label_loss = None
            label_score = None
        axes[index].plot(complexity, loss, linestyle = '-', color = 'r', alpha = 0.6, label = label_loss)
        axes[index].plot(complexity, score, linestyle = '--', color = 'r', alpha = 0.6, label = label_score)
    
    axes[1].legend()
    ax1.set_title('PINN loss')
    plt.tight_layout()
    plt.savefig('figs/statistics_real_appendix.pdf') 

if __name__ == '__main__':
    plot_loss_sim()
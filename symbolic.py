#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:12:11 2022

@author: zzhang99
"""
from net import multi_fnn_index, multi_fnn_index_fixed_bd_norm
from data import generate_data
from pysr import PySRRegressor
import argparse
import os
import numpy as np
from utils import reverse
import pickle

def main(args):
    index = args.index
    weight = 0
    niterations = 200
    maxsize = 12
    c = np.linspace(0,1,100)
    c = np.tile(c[None,:,None], (num_samples,1,1))
    f = fnn_f(f_params, c, indices)
    reverse_indices = reverse(indices)
    model = PySRRegressor(
        model_selection="best",
        niterations=niterations,
        maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
        "exp",
        ],
        complexity_of_operators={"/": 2, "exp": 2},
        loss="loss(x, y, w) = w*(x - y)^2",
        equation_file = f"models/{path}_{index}.csv",
        random_state=0,
        deterministic=True,
        procs=0
    )
    
    c = np.concatenate([c[np.array(reverse_indices[index])][0], [[0], [1]]])
    f = np.concatenate([f[np.array(reverse_indices[index])][0], [[0], [0]]])
    weights = np.concatenate([np.ones(len(c) - 2), [weight, weight]])
    model.fit(c, f, weights = weights)
    pred_str = str(model.sympy())
    print(pred_str)
    with open(f'expressions/{path}_{index}.txt', 'w') as f:
        f.write(pred_str + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='symbolic')
    parser.add_argument('--index', default=0, type=int, help='index of patient')
    parser.add_argument('--T', default=2, type=int, help='time of last data point')
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    args = parser.parse_args()
    
    path = f'{args.T}_{args.seed}'
    os.makedirs('models', exist_ok = True)
    os.makedirs('expressions', exist_ok = True)
    os.makedirs('figs', exist_ok = True)
    fixed_bd = True
    if fixed_bd:
        fnn_f = multi_fnn_index_fixed_bd_norm
    else:
        fnn_f = multi_fnn_index
    t, t_dense, t_extra, c_data, c_data_dense, c_data_extra, mask, num_samples, num_regions, num_groups, L, alpha_data, kappa_data, indices = generate_data(args.T)
    c_params, alpha_params, f_params, kappa_params = pickle.load(open(f'saved/params_{path}', 'rb'))
    main(args)
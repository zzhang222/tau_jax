#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:12:11 2022

@author: zzhang99
"""
from net import multi_fnn_index, multi_fnn_fixed_bd
from example_jax import generate_data
from pysr import PySRRegressor
import argparse
import os
import numpy as np

def main(args):
    index = args.index
    #c = c_data.reshape([c_data.shape[0], -1, 1])
    c = np.linspace(0,1,100)
    c = np.tile(c[None,:,None], (num_samples,1,1))
    f = fnn_f(f_params, c, indices)
    reverse_indices = {}
    for i in range(num_samples):
        reverse_indices[indices[i]] = reverse_indices.get(indices[i], []) + [i]
    reverse_indices = {key: np.array(value) for key, value in reverse_indices.items()}
    
    model = PySRRegressor(
        model_selection="best",
        niterations=300,
        maxsize=14,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
        "exp",
        ],
        complexity_of_operators={"/": 2, "exp": 3},
        loss="loss(x, y, w) = w*(x - y)^2",
        equation_file = f"models/{path}_{index}.csv",
    )
    
    c = np.concatenate([c[np.array(reverse_indices[index])][0], [[0], [1]]])
    f = np.concatenate([f[np.array(reverse_indices[index])][0], [[0], [0]]])
    weights = np.concatenate([np.ones(100), [10, 10]])
    if args.data_type == 'sim':
        alpha = alpha_params.squeeze()[reverse_indices[index]]
        alpha_d = alpha_data.squeeze()[reverse_indices[index]]
        scale = np.mean(alpha)/np.mean(alpha_d)
    else:
        alpha = alpha_params.squeeze()[reverse_indices[index]]
        scale = 1 / np.max(f) / 4
    model.fit(c, f * scale, weights = weights)
    model = PySRRegressor.from_file(f"models/{path}_{index}.pkl")
    pred_str = str(model.sympy())
    print(pred_str)
    with open(f'expressions/{path}_{index}.txt', 'w') as f:
        f.write(pred_str + '\n')
    np.savetxt(f'expressions/scale_{path}_{index}.txt', np.array([scale]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='symbolic')
    parser.add_argument('--index', default=0, type=int, help='index of patient')
    parser.add_argument('--T', default=2, type=int, help='time of last data point')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--data_type', default='real', type=str, help='data type')
    args = parser.parse_args()
    
    path = f'{args.data_type}_{args.T}_{args.seed}'
    os.makedirs('models', exist_ok = True)
    os.makedirs('alpha', exist_ok = True)
    os.makedirs('expressions', exist_ok = True)
    if args.data_type == 'sim':
        fnn_f = multi_fnn_fixed_bd
    else:
        fnn_f = multi_fnn_fixed_bd
    t, t_dense, t_extra, c_data, c_data_dense, c_data_extra, mask, num_samples, num_regions, num_groups, L, alpha_data, kappa_data, indices = generate_data(args.data_type, args.T)
    c_params, alpha_params, f_params, kappa_params = np.load(f'saved/params_{path}.npy', allow_pickle = True)
    main(args)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:40:35 2022

@author: zzhang99
"""
import jax
import numpy as np
import scipy.io as sio
from scipy.integrate import odeint
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.example_libraries import optimizers
import jaxopt
import pandas as pd
from time import time
import argparse
from net import init_network_params, multi_fnn, multi_fnn_index, multi_fnn_fixed_bd, jvp
import os
from jax import config
config.update("jax_enable_x64", True) # To ensure reproducibility

def mse(x, y):
    return ((x - y) ** 2).sum() / x.size

def criterion(c_params, alpha_params, f_params, kappa_params):
    c = multi_fnn(c_params, t)[..., mask]
    c_dense = multi_fnn(c_params, t_dense)
    dataLoss = mse(c_data[..., mask], c)
    dcdt = jvp(c_params, multi_fnn, t_dense)
    fc = - jnp.exp(kappa_params) * c_dense @ L
    c_dense = c_dense.reshape(num_samples, -1, 1)
    fc = fc + alpha_params * fnn_f(f_params, c_dense, indices).reshape(num_samples, -1, num_regions)
    odeLoss = mse(dcdt, fc)
    dfdx = jvp(f_params, fnn_f, c_dense, indices).reshape(num_samples, -1, num_regions)
    resLoss = jax.nn.relu(dfdx - dfdx[:,0:1]).mean()
    return odeLoss + dataLoss + 4 * resLoss

def generate_data(data_type = 'sim', T = -1, T_ext = 3):
    key = random.PRNGKey(321)
    keys = random.split(key, num = 3)
    data = sio.loadmat("tau.mat")
    cort_nodes_df = pd.read_csv('cortical_regions.txt', header = None)
    L = data["L"]
    cort_nodes = cort_nodes_df.values-1
    cort_nodes = cort_nodes[:,0]
    #c0 = random.uniform(keys[0], data['c'][:,0].T.shape)
    c0 = data['c'][:,0].T
    mean_c0, std_c0 = np.mean(c0, axis = 0, keepdims = True), np.std(c0, axis = 0, keepdims = True)
    lower_c0, upper_c0 = -mean_c0/std_c0, (1-mean_c0)/std_c0
    c0 = random.truncated_normal(keys[0], lower_c0, upper_c0, c0.shape) * std_c0 + mean_c0
    num_samples, num_regions = c0.shape
    if data_type == 'sim':
        t = np.linspace(0, T*np.ones(num_samples), T+1).T
    else:
        t = data['t'][:num_samples,:3] - data['t'][:num_samples,:1]
    t_dense = np.linspace(0, t[:,-1], 100).T
    t_extra = np.linspace(0, np.ones(num_samples)*T_ext, 100).T
    
    if data_type == 'sim':
        num_groups = 4
        indices = np.arange(num_samples) % 4
        alpha_noise, alpha_meta_noise, kappa_noise = 0.2, 0.1, 0.7
        alpha_mean, kappa_mean = 0.8, 1.3
        kappa_lower, kappa_upper = -kappa_mean/kappa_noise, float('inf')
        alpha_meta = alpha_mean + alpha_meta_noise * random.normal(keys[0], (num_groups,))[indices]
        alpha = alpha_meta + alpha_noise * random.normal(keys[1], (num_samples,))
        kappa = kappa_mean + kappa_noise * random.truncated_normal(keys[2], kappa_lower, kappa_upper, (num_samples,))
        c = np.array(list(map(lambda i: odeint(g, c0[i], t[i], args=(kappa[i], alpha[i], indices[i], L)), range(num_samples))))
        c_dense = np.array(list(map(lambda i: odeint(g, c0[i], t_dense[i], args=(kappa[i], alpha[i], indices[i], L)), range(num_samples))))
        c_extra = np.array(list(map(lambda i: odeint(g, c0[i], t_extra[i], args=(kappa[i], alpha[i], indices[i], L)), range(num_samples))))
    elif data_type == 'real':
        num_groups = 2
        indices = data['label'][0]
        c = jnp.transpose(data["c"][:,:3], (2,1,0))
        c_dense = c_extra = alpha = kappa = -1
    t, t_dense, t_extra = t[..., None], t_dense[..., None], t_extra[..., None]
    return t, t_dense, t_extra, c, c_dense, c_extra, cort_nodes, num_samples, num_regions, num_groups, L, alpha, kappa, indices

def reaction(c, alpha, index):
    if index == 0:
        return alpha * c * (1 - c)
    elif index == 1:
        return alpha * c * (1 - c ** 2)
    elif index == 2:
        return alpha * (c ** 2 - c ** 3)
    elif index == 3:
        return alpha * c * (1 - c) * np.exp(c - 1)
    else:
        raise NotImplementedError
        
def g(c, t, kappa, alpha, index, L):
    return - kappa * L @ c + reaction(c, alpha, index)

def callback(x):
    loss_lbfgs.append(criterion_lbfgs(x))

@jit
def update(step_i, opt_state):
    c_params, alpha_params, f_params, kappa_params = get_params(opt_state)
    loss, grads = value_and_grad(criterion, argnums = (0,1,2,3))(c_params, alpha_params, f_params, kappa_params)
    opt_state = opt_update(step_i, grads, opt_state)
    return loss, opt_state

def train(opt_state, steps, print_every):
    t1 = time()
    for step_i in range(steps+1):
        loss, opt_state = update(step_i, opt_state)
        if step_i % print_every == 0:
            print('Step: ' + str(step_i) + ', Loss: ' + str(loss))
            loss_adam.append(loss)
    t2 = time()
    params, state = solver.run(get_params(opt_state))
    print(state.fun_val, loss)
    t3 = time()
    print('Adam time: ' + str(t2 - t1))
    print('LBFGS time: ' + str(t3 - t2))
    return params

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PINN')
    parser.add_argument('--data_type', default='real', type=str, help='sim or real')
    parser.add_argument('--seed', default=321, type=int, help='random seed')
    parser.add_argument('--T', default=2, type=int, help='time of last data point')
    args = parser.parse_args()
    os.makedirs('saved', exist_ok = True)
    
    seed = args.seed
    key = random.PRNGKey(seed)
    data_type = args.data_type
    T = args.T
    t, t_dense, t_extra, c_data, c_data_dense, c_data_extra, mask, num_samples, num_regions, num_groups, L, alpha_data, kappa_data, indices = generate_data(data_type, T)
    layers = 3
    width = 100
    print_every = 1000
    debug = False
    path = f'{data_type}_{T}_{seed}'
    loss_adam, loss_lbfgs = [], []
    if data_type == 'sim':
        fnn_f = multi_fnn_fixed_bd
    else:
        fnn_f = multi_fnn_fixed_bd
    if debug:
        steps = 0
        c_params, alpha_params, f_params, kappa_params = jnp.load(f'saved/params_{path}.npy', allow_pickle = True)
    else:
        steps = 100000
        keys = random.split(key)
        c_params = init_network_params(1, num_regions, width, num_samples, keys[0], layers)
        alpha_params = jnp.zeros((num_samples, 1, 1))
        f_params = init_network_params(1, 1, 20, num_groups, keys[1], 3)
        kappa_params = jnp.zeros((num_samples, 1, 1))
    
    opt_init, opt_update, get_params = optimizers.adam(1e-3)
    opt_state = opt_init((c_params, alpha_params, f_params, kappa_params))
    criterion_lbfgs = jit(lambda x: criterion(*x))
    solver = jaxopt.ScipyMinimize(fun = criterion_lbfgs, maxiter = 50000,
                                  method = 'L-BFGS-B', tol = 0, options = {'iprint':1}, callback = callback)
    params = train(opt_state, steps, print_every)
    jnp.save(f'saved/params_{path}', params, allow_pickle = True)
    loss_adam, loss_lbfgs = np.array(loss_adam), np.array(loss_lbfgs)
    np.save(f'saved/loss_adam_{path}', loss_adam)
    np.save(f'saved/loss_lbfgs_{path}', loss_lbfgs)
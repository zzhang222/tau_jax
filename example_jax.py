#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:40:35 2022

@author: zzhang99
"""
import jax
import numpy as np
import jax.numpy as jnp
from jax import random, jit, value_and_grad
from jax.example_libraries import optimizers
import jaxopt
from time import time
import argparse
from net import init_network_params, multi_fnn, multi_fnn_index, multi_fnn_index_fixed_bd_norm
from utils import mse, jvp
from data import generate_data
import os
import pickle
#from jax import config
#config.update("jax_enable_x64", True) # double digit more stable (always same result for the same random seeds)

def criterion(c_params, alpha_params, f_params, kappa_params):
    c = multi_fnn(c_params, t)[..., mask]
    c_dense = multi_fnn(c_params, t_dense)
    dataLoss = mse(c_data[..., mask], c)
    dcdt = jvp(c_params, multi_fnn, t_dense)
    fc = - jnp.exp(kappa_params) * c_dense @ L
    c_dense = c_dense.reshape(num_samples, -1, 1)
    fc = fc + alpha_params * fnn_f(f_params, c_dense, indices).reshape(num_samples, -1, num_regions)
    odeLoss = mse(dcdt, fc)
    c_aux = jnp.tile(jnp.linspace(0, 1, 100)[None, :, None], (c_dense.shape[0], 1, 1))
    dfdx = jvp(f_params, fnn_f, c_aux, indices) * alpha_params
    dfdx0 = jvp(f_params, fnn_f, jnp.zeros_like(c_aux), indices) * alpha_params
    resLoss = (jax.nn.relu(dfdx - dfdx0)).mean()
    return odeLoss + dataLoss + lam * resLoss

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
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    parser.add_argument('--T', default=2, type=int, help='time of last data point')
    args = parser.parse_args()
    os.makedirs('saved', exist_ok = True)
    
    seed = args.seed
    key = random.PRNGKey(seed)
    T = args.T
    t, t_dense, t_extra, c_data, c_data_dense, c_data_extra, mask, num_samples, num_regions, num_groups, L, alpha_data, kappa_data, indices = generate_data(T)
    c_layers = 3
    c_width = 100
    f_layers = 3
    f_width = 10
    print_every = 10
    lam = 1
    lbfgs_steps = 50000
    debug = False
    fixed_bd = True
    path = f'{T}_{seed}'
    loss_adam, loss_lbfgs = [], []
    if fixed_bd:
        fnn_f = multi_fnn_index_fixed_bd_norm
    else:
        fnn_f = multi_fnn_index
    if debug:
        steps = 0
        c_params, alpha_params, f_params, kappa_params = pickle.load(open(f'saved/params_{path}', 'rb'))
    else:
        steps = 100000
        keys = random.split(key)
        c_params = init_network_params(1, num_regions, c_width, num_samples, keys[0], c_layers)
        alpha_params = jnp.zeros((num_samples, 1, 1))
        f_params = init_network_params(1, 1, f_width, num_groups, keys[1], f_layers)
        kappa_params = jnp.zeros((num_samples, 1, 1))
    
    opt_init, opt_update, get_params = optimizers.adam(1e-3)
    opt_state = opt_init((c_params, alpha_params, f_params, kappa_params))
    criterion_lbfgs = jit(lambda x: criterion(*x))
    solver = jaxopt.ScipyMinimize(fun = criterion_lbfgs, maxiter = lbfgs_steps,
                                  method = 'L-BFGS-B', tol = 0, options = {'iprint':1}, callback = callback)
    params = train(opt_state, steps, print_every)
    pickle.dump(params, open(f'saved/params_{path}', "wb"))
    loss_adam, loss_lbfgs = np.array(loss_adam), np.array(loss_lbfgs)
    np.save(f'saved/loss_adam_{path}', loss_adam)
    np.save(f'saved/loss_lbfgs_{path}', loss_lbfgs)

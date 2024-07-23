#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 22:13:34 2022

@author: zzhang99
"""
from jax import random
import jax.numpy as jnp
import jax.nn as nn

def random_layer_params(ind, outd, numNN, key):
    scale = jnp.sqrt(2 / (ind + outd))
    return random.normal(key, (numNN, ind, outd)) * scale, jnp.zeros((numNN, 1, outd))

def init_network_params(ind, outd, width, numNN, key, layers):
    keys = random.split(key, layers)
    layers = [random_layer_params(ind, width, numNN, keys[0])]
    layers += [random_layer_params(width, width, numNN, key) for key in keys[1:-1]]
    layers += [random_layer_params(width, outd, numNN, keys[-1])]
    return layers

def multi_fnn_index(params, x, indices = None):
    for i in range(len(params) - 1):
        W, b = params[i]
        W, b = W[indices], b[indices]
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    W, b = W[indices], b[indices]
    x = x @ W + b
    return x

def multi_fnn_index_fixed_bd(params, x, indices = None):
    bd_term = x * (1 - x)
    x = jnp.exp(multi_fnn_index(params, x, indices)) * bd_term
    return x

    
def multi_fnn_index_fixed_bd_norm(params, x, indices = None):
    f = multi_fnn_index_fixed_bd(params, x, indices)
    x_all = jnp.tile(jnp.linspace(0, 1, 100)[None, :, None], (x.shape[0], 1, 1))
    f_max = multi_fnn_index_fixed_bd(params, x_all, indices).max(axis=1, keepdims=True)
    return f / f_max / 4

def multi_fnn(params, x):
    for i in range(len(params) - 1):
        W, b = params[i]
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    x = x @ W + b
    return x
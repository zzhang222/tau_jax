#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 22:13:34 2022

@author: zzhang99
"""
from jax import random, jacrev
import jax.numpy as jnp
import jax

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

def multi_fnn_fixed_bd(params, x, indices = None):
    bd_term = x * (1 - x)
    for i in range(len(params) - 1):
        W, b = params[i]
        W, b = W[indices], b[indices]
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    W, b = W[indices], b[indices]
    x = jnp.exp(x @ W + b) * bd_term
    return x
    
def multi_fnn(params, x):
    for i in range(len(params) - 1):
        W, b = params[i]
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    x = x @ W + b
    return x

def fnn_sum(params, f, x, *args):
    return jnp.sum(f(params, x, *args), axis=(0, 1))

def fnn_jacobian(params, f, x, *args):
    dfdx = jacrev(fnn_sum, argnums = 2)(params, f, x, *args).squeeze(-1)
    return jnp.transpose(dfdx, (1,2,0))

def jvp(params, f, x, *args):
    # make a JAX function
    def _fn(_x):
        return f(params, _x, *args)
    
    # call jvp (forward mode) for computation of gradients
    _, tangents = jax.jvp(_fn, primals=(x, ), tangents=(jnp.ones_like(x), ))
    return tangents
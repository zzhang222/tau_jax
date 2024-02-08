import matplotlib.pyplot as plt
from net import multi_fnn_index, multi_fnn, multi_fnn_fixed_bd
import os
from example_jax import generate_data
import numpy as np
from scipy.integrate import odeint
from pysr import PySRRegressor
from jax import jit
import jax.numpy as jnp

def f_symbolic(c, t, i, model, kappa, alpha):
    f = model.predict(c[:, None])
    return - kappa * L @ c + alpha.squeeze() * f

def compute_error():
    c_symbolic_negative, c_symbolic_positive = [], []
    for i in range(num_samples):
        c_symbolic = []
        for seed in seeds:
            path = f'{data_type}_{T}_{seed}'
            model = PySRRegressor.from_file(f"models/{path}_{indices[i]}.pkl")
            c_symbolic.append(odeint(f_symbolic, c_data[i, 0], t[i].squeeze(), args=(i, model, kappa_params[seed-1][i], alpha_params[seed-1][i])))
        c_symbolic = np.array(c_symbolic)
        if indices[i] == 0:
            c_symbolic_negative.append(c_symbolic)
        else:
            c_symbolic_positive.append(c_symbolic)
    c_symbolic_negative = np.array(c_symbolic_negative)
    c_symbolic_positive = np.array(c_symbolic_positive)
    error_negative = np.mean((c_symbolic_negative[..., mask] - c_data_negative[..., mask]) ** 2, axis = (0, 2, 3))
    error_positive = np.mean((c_symbolic_positive[..., mask] - c_data_positive[..., mask]) ** 2, axis = (0, 2, 3))
    print(error_negative, error_positive)
    np.savetxt(f'expressions/error_negative.txt', error_negative)
    np.savetxt(f'expressions/error_positive.txt', error_positive)

if __name__ == '__main__':
    data_type = 'real'
    T = 2
    t, t_dense, t_extra, c_data, c_data_dense, c_data_extra, mask, num_samples, num_regions, num_groups, L, alpha_data, kappa_data, indices = generate_data(data_type, T, 30)
    fnn_f = multi_fnn_fixed_bd
    reverse_indices = {}
    for i in range(num_samples):
        reverse_indices[indices[i]] = reverse_indices.get(indices[i], []) + [i]
    reverse_indices = {key: np.array(value) for key, value in reverse_indices.items()}
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
    c_data_negative = c_data[indices == 0, None]
    c_data_positive = c_data[indices == 1, None]
    print(c_data_positive.shape, c_data_negative.shape)
    compute_error()
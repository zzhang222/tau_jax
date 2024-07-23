import jax.numpy as jnp
import jax
import numpy as np

def mse(x, y):
    return ((x - y) ** 2).sum() / x.size

def jvp(params, f, x, *args):
    # make a JAX function
    def _fn(_x):
        return f(params, _x, *args)
    
    # call jvp (forward mode) for computation of gradients
    _, tangents = jax.jvp(_fn, primals=(x, ), tangents=(jnp.ones_like(x), ))
    return tangents

def parse_region_index(region_name):
    with open('tau_data/nodes_regionName.txt') as f:
        name_list = f.readlines()
    region_names = np.array([region_name in name for name in name_list])
    return region_names.nonzero()[0]

def get_min_max(x):
    return np.min(x, axis=0), np.max(x, axis=0)

def get_mean_std(x):
    return np.mean(x, axis=0), np.std(x, axis=0)

def reverse(indices):
    reverse_indices = {}
    for i in range(len(indices)):
        reverse_indices[indices[i]] = reverse_indices.get(indices[i], []) + [i]
    reverse_indices = {key: np.array(value) for key, value in reverse_indices.items()}
    return reverse_indices
"""
This module wraps around the ytopt generator.
"""
import numpy as np

__all__ = ['uniform_random_sample',
           'uniform_random_sample_with_variable_resources',
           'uniform_random_sample_obj_components',
           'latin_hypercube_sample',
           'uniform_random_sample_cancel']


def latin_hypercube_sample(H, persis_info, gen_specs, _):

    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']

    n = len(lb)
    b = gen_specs['user']['gen_batch_size']

    H_o = np.zeros(b, dtype=gen_specs['out'])

    A = lhs_sample(n, b, persis_info['rand_stream'])

    # H_o['BLOCK_SIZE'] = np.ceil(A*(ub-lb)+lb)
    H_o['BLOCK_SIZE'] = np.arange(1,11)

    return H_o, persis_info


def lhs_sample(n, k, stream):

    # Generate the intervals and random values
    intervals = np.linspace(0, 1, k+1)
    rand_source = stream.uniform(0, 1, (k, n))
    rand_pts = np.zeros((k, n))
    sample = np.zeros((k, n))

    # Add a point uniformly in each interval
    a = intervals[:k]
    b = intervals[1:]
    for j in range(n):
        rand_pts[:, j] = rand_source[:, j]*(b-a) + a

    # Randomly perturb
    for j in range(n):
        sample[:, j] = rand_pts[stream.permutation(k), j]

    return sample

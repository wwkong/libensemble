"""
This module is a wrapper around an example ytopt objective function
"""
__all__ = ['one_d_example']

import numpy as np
# from autotune import TuningProblem
# from autotune.space import *
# import os, sys, time, json, math
# import ConfigSpace as CS
# import ConfigSpace.hyperparameters as CSH
# from skopt.space import Real, Integer, Categorical

from plopper import Plopper


def one_d_example(x, persis_info, sim_specs, libE_info):
    y = myobj({'BLOCK_SIZE': np.squeeze(x['BLOCK_SIZE'])}, libE_info['workerID'])  # ytopt objective wants a dict
    H_o = np.zeros(1, dtype=sim_specs['out'])
    H_o['f'] = y

    return H_o, persis_info


obj = Plopper('./mmm_block.cpp', './')


def myobj(point: dict, workerID):
    def plopper_func(value):
        params = ['BLOCK_SIZE']
        result = obj.findRuntime(value, params, workerID)
        return result

    x = np.array([point['BLOCK_SIZE']])
    results = plopper_func(x)
    print('CONFIG and OUTPUT', [point, results], flush=True)
    return results

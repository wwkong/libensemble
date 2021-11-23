"""
This module wraps around the ytopt generator.
"""
import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = ['persistent_ytopt']


def persistent_ytopt(H, persis_info, gen_specs, libE_info):

    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(lb)
    b = gen_specs['user']['gen_batch_size']
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs['out'])
        H_o['BLOCK_SIZE'] = persis_info['rand_stream'].uniform(lb, ub, (b, n))
        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, '__len__'):
            b = len(calc_in)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG

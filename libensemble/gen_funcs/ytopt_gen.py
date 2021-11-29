"""
This module wraps around the ytopt generator.
"""
import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport


__all__ = ['persistent_ytopt']


def persistent_ytopt(H, persis_info, gen_specs, libE_info):

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    user_specs = gen_specs['user']
    ytoptimizer = user_specs['ytoptimizer']

    # Send batches until manager sends stop tag
    tag = None
    calc_in = None
    first_call = True
    while tag not in [STOP_TAG, PERSIS_STOP]:

        if first_call:
            XX = ytoptimizer.ask_initial(n_points=user_specs['num_sim_workers'])  # Returns a list
            batch_size = len(XX)
            first_call = False
        else:
            # The hand-off of information from libE to ytopt is below. This hand-off may be brittle.
            batch_size = len(calc_in)
            results = []
            for entry in calc_in:
                results += [({'BLOCK_SIZE': entry['BLOCK_SIZE'][0]}, entry['RUN_TIME'])]

            ytoptimizer.tell(results)

            XX = ytoptimizer.ask(n_points=batch_size)  # Returns a generator that we convert to a list
            XX = list(XX)[0]

        # The hand-off of information from ytopt to libE is below. This hand-off may be brittle.
        H_o = np.zeros(batch_size, dtype=gen_specs['out'])
        for i, entry in enumerate(XX):
            for key, value in entry.items():
                H_o[i][key] = value

        # print('requesting:', H_o['BLOCK_SIZE'], flush=True)
        tag, Work, calc_in = ps.send_recv(H_o)
        # print('received:', calc_in, flush=True)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG

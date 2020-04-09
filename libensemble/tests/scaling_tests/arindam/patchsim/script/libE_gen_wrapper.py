"""
Generator wrapper
"""
import sys
from run_mcmc import main
from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG

def mcmc_wrapper(H, persis_info, gen_specs, libE_info):

    U = gen_specs['user']
    gt_datapath = U['gt_datapath']
    state = U['state']
    patch_input_datadir = U['patch_input_datadir']
    nsamp = U['nsamp']
    
    pout = main(gt_datapath, state, patch_input_datadir, nsamp, gen_specs, libE_info)

    persis_info['pout'] = pout

    return [], persis_info, FINISHED_PERSISTENT_GEN_TAG

"""
Generator wrapper
"""
from run_mcmc import main
from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG
from argparse import Namespace

def mcmc_wrapper(H, persis_info, gen_specs, libE_info):

    d = Namespace(**gen_specs['user'])

    pout = main(d, gen_specs, libE_info)

    persis_info['pout'] = pout

    return [], persis_info, FINISHED_PERSISTENT_GEN_TAG

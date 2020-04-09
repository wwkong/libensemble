import numpy as np
import sys
from run_mcmc import get_LL
from argparse import Namespace  


def likelihood(H, persis_info, sim_specs, _):
    """
    Evaluates likelihood
    """
    d = Namespace(**sim_specs['user']) 
    d.params['alpha'] = H['alpha'][0]
    d.params['beta'] = H['beta'][0]
    d.params['gamma'] = H['gamma'][0]

    H_o = np.zeros(1, dtype=sim_specs['out'])
    H_o['f'][0] = get_LL(d.configs, d.patch_df, d.params, d.Theta, d.seeds, d.vaxs, d.gt_va, d.cov_mat_dict, d.gt_FIPS, random_seed=None)
    print( H_o['f'][0] )
    
    return H_o, persis_info

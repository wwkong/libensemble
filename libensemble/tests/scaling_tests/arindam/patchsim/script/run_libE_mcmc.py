#!/usr/bin/env python
# Execute via 
#    mpiexec -np 3 python3 run_libE_mcmc.py
import numpy as np

# Import libEnsemble modules
from libensemble.libE import libE
from libE_sim_wrapper import likelihood as sim_f
from libE_gen_wrapper import mcmc_wrapper as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

from run_mcmc import get_state_gt, setupModel, get_cov
nworkers, is_master, libE_specs, _ = parse_args()


if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

# Initialize values
gt_datapath = "../input/us-counties.csv"
state = 'Virginia'
patch_input_datadir = "/home/jlarson/research/libensemble/libensemble/tests/scaling_tests/arindam/patchsim/input/"

gt_va = get_state_gt(gt_datapath, state)
configs, patch_df, params, seeds, Theta, vaxs = setupModel(patch_input_datadir)
cov_mat_dict = get_cov(gt_va)
gt_FIPS = np.array(gt_va.columns)

# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': sim_f,           # Function whose output is being minimized
             'in': ['alpha', 'beta', 'gamma'],  # Name of input for sim_f
             'out': [('f', float)],          # Objective to be minimized
             'user': {'configs': configs,
                      'patch_df': patch_df, 
                      'Theta': Theta,   
                      'params': params,  # Will be partially overwritten in sim_f
                      'seeds': seeds,   
                      'vaxs': vaxs, 
                      'gt_va': gt_va,   
                      'cov_mat_dict': cov_mat_dict, 
                      'gt_FIPS': gt_FIPS}
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': gen_f,                 # Generator function
             'in': [],                       # Generator input
             'out': [('alpha', float), ('beta', float, (133,190)), ('gamma', float)],       # nb of parameters to input into sim
             'user': {'gt_datapath': gt_datapath,
                      'state': state,
                      'patch_input_datadir': patch_input_datadir,
                      'nsamp': 10}

             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}

# Create a different random number stream for each worker and the manager
persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'elapsed_wallclock_time': 10000} # Just setting a large time, as the gen_f will stop producing points when it's done

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

# Save results to numpy file
if is_master:
    save_libE_output(H, persis_info, __file__, nworkers)

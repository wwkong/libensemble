"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python3 test_1d_sampling.py
   python3 test_1d_sampling.py --nworkers 3 --comms local
   python3 test_1d_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.ytopt_obj import one_d_example as sim_f
from libensemble.gen_funcs.ytopt_gen import latin_hypercube_sample as gen_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

nworkers, is_manager, libE_specs, _ = parse_args()

sim_specs = {
    'sim_f': sim_f,
    'in': ['BLOCK_SIZE'],
    'out': [('f', float)],
}

gen_specs = {
    'gen_f': gen_f,
    'out': [('BLOCK_SIZE', int, (1,) )],
    'user': {
        'gen_batch_size': 10,
        'lb': np.array([1]),
        'ub': np.array([10]),
    },
}

persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

exit_criteria = {'sim_max': 10}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

if is_manager:
    assert len(H) == exit_criteria['sim_max']
    print("\nlibEnsemble has perform the correct number of evaluations")
    save_libE_output(H, persis_info, __file__, nworkers)

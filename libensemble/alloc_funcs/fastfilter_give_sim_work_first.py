import numpy as np

from libensemble.tools.alloc_support import avail_worker_ids, sim_work, gen_work, count_gens


def give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    Decide what should be given to workers. This allocation function gives any
    available simulation work first, and only when all simulations are
    completed or running does it start (at most ``alloc_specs['user']['num_active_gens']``)
    generator instances.

    Allows for a ``alloc_specs['user']['batch_mode']`` where no generation
    work is given out unless all entries in ``H`` are returned.

    Allows for ``blocking`` of workers that are not active, for example, so
    their resources can be used for a different simulation evaluation.

    Can give points in highest priority, if ``'priority'`` is a field in ``H``.

    This is the default allocation function if one is not defined.

    .. seealso::
        `test_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling.py>`_ # noqa
    """

    low_bound = persis_info.get('low_bound',0)

    #Make low_bound do nothing while i test
    H = H[low_bound:]

    # Separate low_bounds for given/returned/given_back or single low_bound?
    # low_bound may be passed through to give_sim_work_first but prob want to set in allocation function.
    #   and use via either function attribute or persis_info. Alloc function should decide if can ignore points.

    # if send all values so far to gen (not great!) but if do - still need all H.

    Work = {}
    gen_count = count_gens(W)
    avail_set = set(W['worker_id'][np.logical_and(~W['blocked'],
                                                  W['active'] == 0)])

    import pdb;pdb.set_trace()

    task_avail = ~H['given']
    for i in avail_worker_ids(W):

        if i not in avail_set:
            pass

        elif np.any(task_avail):
        #elif not np.all(H['allocated']):

            # Pick all high priority, oldest high priority, or just oldest point
            if 'priority' in H.dtype.fields:
                priorities = H['priority'][task_avail]
                if gen_specs['user'].get('give_all_with_same_priority'):
                    q_inds = (priorities == np.max(priorities))
                else:
                    q_inds = np.argmax(priorities)
                # As below - this is wrong as q_inds is  but change not allocated stuff
                #while low_bound in q_inds:  # Again index v sim_id
                    #low_bound += 1
            else:
                q_inds = 0


            # Get sim ids and check resources needed
            sim_ids_to_send = np.nonzero(task_avail)[0][q_inds] + low_bound

            #SH These are not sim_ids (but indices!) - usually the same - but should test not being the same.
            # H_rows = np.nonzero(task_avail)[0][q_inds]  # intuitive enough?

            sim_ids_to_send = np.atleast_1d(sim_ids_to_send)
            nodes_needed = (np.max(H[sim_ids_to_send]['num_nodes'])
                            if 'num_nodes' in H.dtype.names else 1)
            if nodes_needed > len(avail_set):
                break

            # Assign resources and mark tasks as allocated to workers
            sim_work(Work, i, sim_specs['in'], sim_ids_to_send, persis_info.get(i))
            #H['allocated'][sim_ids_to_send] = True
            task_avail[sim_ids_to_send] = False

            for task in task_avail: # use func - inline with task_avail.
                if task:
                    break
                low_bound += 1

            print('sim_ids: {}   New low bound: {}'.format(sim_ids_to_send, low_bound))
            import pdb;pdb.set_trace()


            # Update resource records
            avail_set.remove(i)
            if nodes_needed > 1:
                workers_to_block = list(avail_set)[:nodes_needed-1]
                avail_set.difference_update(workers_to_block)
                Work[i]['libE_info']['blocking'] = workers_to_block

        else:

            # Allow at most num_active_gens active generator instances
            if gen_count >= alloc_specs['user'].get('num_active_gens', gen_count+1):
                break

            # No gen instances in batch mode if workers still working
            still_working = ~H['returned']  ## SH working if given but not returned??? but okay.
            if alloc_specs['user'].get('batch_mode') and np.any(still_working):
                break

            # Give gen work
            gen_count += 1
            if 'in' in gen_specs and len(gen_specs['in']):
                gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info.get(i))
            else:
                gen_work(Work, i, [], [], persis_info.get(i))

    persis_info['low_bound'] = low_bound
    return Work, persis_info

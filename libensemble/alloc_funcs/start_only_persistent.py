import numpy as np
from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.tools.alloc_support import (avail_worker_ids,
                                             sim_work, gen_work,
                                             count_persis_gens,
                                             assign_resources)


# SH TODO: Either replace only_persistent_gens or add a different alloc func (or file?)
#          Check/update docstring
def only_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start up to 1 persistent generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling_async.py>`_ # noqa
    """

    Work = {}
    gen_count = count_persis_gens(W)

    # SH TODO - for testing only
    try:
        only_persistent_gens.counter += 1
    except AttributeError:
        only_persistent_gens.counter = 1
    # print('count',only_persistent_gens.counter)

    if persis_info.get('gen_started') and gen_count == 0:
        # The one persistent worker is done. Exiting
        return Work, persis_info, 1

    for i in avail_worker_ids(W, persistent=EVAL_GEN_TAG):
        if gen_specs['user'].get('async', False):
            # If i is in persistent mode, asynchronous behavior is desired, and
            # *any* of its calculated values have returned, give them back to i.
            # Otherwise, give nothing to i
            returned_but_not_given = np.logical_and.reduce((H['returned'], ~H['given_back'], H['gen_worker'] == i))
            if np.any(returned_but_not_given):
                inds_to_give = np.where(returned_but_not_given)[0]
                gen_work(Work, i,
                         sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')],
                         np.atleast_1d(inds_to_give), persis_info[i], persistent=True)

                H['given_back'][inds_to_give] = True

        else:
            # If i is in persistent mode, batch behavior is desired, and
            # *all* of its calculated values have returned, give them back to i.
            # Otherwise, give nothing to i
            gen_inds = (H['gen_worker'] == i)
            if np.all(H['returned'][gen_inds]):
                last_time_gen_gave_batch = np.max(H['gen_time'][gen_inds])
                inds_to_give = H['sim_id'][gen_inds][H['gen_time'][gen_inds] == last_time_gen_gave_batch]
                gen_work(Work, i,
                         sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')],
                         np.atleast_1d(inds_to_give), persis_info[i], persistent=True)

                H['given_back'][inds_to_give] = True

    task_avail = ~H['given']  # SH TODO: Unchanged - but what if allocated space in array that is not genned?

    # SH TODO: Now the give_sim_work_first bit - should merge to avoid duplicating functionality
    avail_workers = avail_worker_ids(W, persistent=False, zero_resource_workers=False)

    while avail_workers:

        if not np.any(task_avail):
            break

        # Perform sim evaluations (if they exist in History).
        sim_ids_to_send = np.nonzero(task_avail)[0][0]  # oldest point

        num_rsets_req = (np.max(H[sim_ids_to_send]['resource_sets'])
                         if 'resource_sets' in H.dtype.names else 1)

        # If more than one group (node) required, allocates whole nodes - also removes from avail_workers
        print('\nrset_team being called for sim. Requesting {} rsets'.format(num_rsets_req))

        rset_team = assign_resources(num_rsets_req, avail_workers[0])
        print('resource team for sim', rset_team, flush=True)

        # print('AFTER ASSIGN sim ({}): avail_workers: {}'.format(rset_team,avail_workers), flush=True)

        # SH TODO: Determine if next few lines could be combined.....

        # None means insufficient available resources for this work unit
        if rset_team is None:
            break

        # SH TODO consider whether to do this is assign_resources
        worker = avail_workers.pop(0)  # Give to first worker in list

        sim_work(Work, worker, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info[worker])

        print('Packed for worker: {}. Resource team for sim: {}\n'.format(worker, rset_team), flush=True)

        task_avail[sim_ids_to_send] = False

        # SH TODO this could be done in sim_work() or maybe all combined in assign_resources
        Work[worker]['libE_info']['rset_team'] = rset_team  # SH TODO: Maybe do in sim_work?

    # A separate loop/section as now need zero_resource_workers for gen.
    # SH TODO   - with rsets -  zero_resource_workers only needed if using fixed worker/resource mapping.
    #             so really then want to say use a zrw, if its set, else use any!!!!
    #             alternatively may be sim/gen assigned workers.
    if not np.any(task_avail):
        avail_workers = avail_worker_ids(W, persistent=False, zero_resource_workers=True)

        while avail_workers:
            # SH TODO: So we don't really need a loop here for this, but general case would allow multiple gens
            if gen_count == 0:
                # Finally, call a persistent generator as there is nothing else to do.
                gen_count += 1

                print('\nrset_team being called for gen')

                # SH TODO: How would you provide resources to a gen? Maybe via persis_info if variable?
                gen_resources = persis_info.get('gen_resources', 0)

                # SH TODO. Dont need to call if required resources is zero
                # Worker is only used if resources are assigned.
                rset_team = assign_resources(gen_resources, avail_workers[0])

                print('resource team for gen', rset_team, flush=True)

                # None means insufficient available resources for this work unit
                if rset_team is None:
                    break

                worker = avail_workers.pop(0)  # Give to first worker in list

                # print('AFTER ASSIGN gen ({}): avail_workers: {}'.format(worker,avail_workers), flush=True)

                gen_work(Work, worker, gen_specs['in'], range(len(H)), persis_info[worker],
                         persistent=True)

                # Even if empty list, presence of rset_team stops manager giving default resources
                Work[worker]['libE_info']['rset_team'] = rset_team

                print('Packed for worker: {}. Resource team for gen: {}\n'.format(worker, rset_team), flush=True)

                persis_info['gen_started'] = True

    return Work, persis_info, 0

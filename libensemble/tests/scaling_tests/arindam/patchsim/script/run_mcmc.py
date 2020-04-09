# from datetime import datetime
# import pprint as pp
# import json
# import sys
# import os
# import matplotlib.dates as mdates
# from functools import partial
import pandas as pd
import numpy as np
import patchsim as sim
import scipy.stats as st

from libensemble.tools.gen_support import sendrecv_mgr_worker_msg
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP


def get_state_gt(datapath, state):
    gt_df = pd.read_csv(datapath, dtype={'fips':str}, parse_dates=['date'])
    gt_df = gt_df[gt_df.state==state][['date','fips','cases']]
    gt_df = gt_df.pivot(index='date',columns='fips',values='cases')
    gt_df.drop(np.nan,axis=1,inplace=True)
    gt_df = gt_df.reindex(pd.date_range(pd.Timestamp('2020-02-23'),gt_df.index.max())).fillna(0)
    return gt_df

def df_shift_scale(df,delay,scale,rounding=True):
    return df.reindex(columns = range(1,df.columns.max()+delay+1)).fillna(0).astype(int).shift(periods=delay,axis='columns').fillna(0.0)*scale

def run_patch(configs, patch_df, params, Theta, seeds, vaxs):
    df = sim.run_disease_simulation(configs, patch_df, params, Theta, seeds, vaxs,
            return_epi=True, write_epi=False, log_to_file=False)
    df.columns = range(len(df.columns))
    return df

def get_patch_outcome(df,outcome='C'):

    symprob = 0.5
    del_I = 5 ## average incubation period

#     scaling = 0.15 ## detection rate from exposure
#     del_C = 8 ## average time to detect from exposure

    scaling = 0.3 ## detection rate from infections
    del_C = 3 ## average time to detect from illness onset

    ##Hospitalization
    p_h = 0.2
    del_h = 5
    # D_h = 5

    ##ICU
    p_u = 0.2
    del_u = 1

    ##Ventilation
    p_v = 0.75
    del_v = 2
    # D_v = 14

    ##Death
    p_d = 0.0025
    del_d = 10

    I = df_shift_scale(df,del_I,symprob).round()
    C = df_shift_scale(I,del_C,scaling).round()

    H = df_shift_scale(I,del_h,p_h).round()
    U = df_shift_scale(H,del_u,p_u).round()
    V = df_shift_scale(U,del_v,p_v).round()

    D = df_shift_scale(I,del_d,p_d).round()

    return eval(outcome)

def get_loglikelihood(gt_va, sim_df, cov_mat_dict, FIPS):
    s = 0
    for fips in FIPS:
        ## subset sim output to match dimension with ground truth
        gt_vec = gt_va.loc[:,fips]
        ll = len(gt_vec)
        gt_fips_vec = gt_vec.to_numpy() + 1
        sim_fips_vec = sim_df.loc[fips,:ll].to_numpy() + 1

        ## now compute the normal pdf
        s = s + st.multivariate_normal.logpdf(np.log(sim_fips_vec.cumsum()),
                mean = np.log(gt_fips_vec), cov = cov_mat_dict[fips])
    return s

def get_LL(configs, patch_df, params, Theta, seeds, vaxs, gt_va, cov_mat_dict, gt_FIPS, random_seed=None):
    if random_seed:
        configs['RandomSeed'] = random_seed
    else:
        try:
            del configs['RandomSeed']
        except:
            pass
    sim_1 = get_patch_outcome(run_patch(configs, patch_df, params, Theta, seeds, vaxs),outcome='C')
    loglik_1 = get_loglikelihood(gt_va, sim_1, cov_mat_dict, gt_FIPS)
    return loglik_1


# prepare the covariances

def get_cov(gt_va, sd_prop=0.5):

    cov_mat_dict = {}
    gt_FIPS = np.array(gt_va.columns)

    c_mat = np.tril(np.repeat(1, gt_va.shape[0]),0)

    for fips in gt_FIPS:
        gt_vec = gt_va.loc[:,fips]
        m = gt_vec.values

        # m is cumulative count, make it incremental first
        m_inc = np.append(m[0], np.diff(m))
        s = m_inc * sd_prop

        # set a threshold for sd
        thresh = 10
        ss = np.array([thresh if x < thresh else x for x in s])

        c_mat_ = c_mat[0:len(m), 0:len(m)]
        cov_mat_native = np.matmul(np.matmul(c_mat_, np.diag(ss)), np.transpose(c_mat_))
        x = np.random.multivariate_normal(m, cov_mat_native, size = 1000)
        x[x<0] = 0.1
        cov_mat_dict[fips] = np.cov(np.transpose(np.log(x)))

    return cov_mat_dict


def setupModel(patch_input_datadir):

    configs = sim.read_config(patch_input_datadir + 'cfg_base')
    del configs['LogFile']
    del configs['OutputFile']
    configs['LoadState'] = 'False'
    configs['SaveState'] = 'False'
    configs['Duration'] = 190 ## end of August 2020

    patch_df = sim.load_patch(configs)
    params = sim.load_params(configs, patch_df)
    seeds = sim.load_seed(configs, params, patch_df)
    Theta = sim.load_Theta(configs, patch_df)
    vaxs = sim.load_vax(configs, params, patch_df)

    return configs, patch_df, params, seeds, Theta, vaxs


def run_mcmc(configs, patch_df, params, seeds, Theta, vaxs, gt_va,
        cov_mat_dict, nsamp, gen_specs, libE_info):

    gt_FIPS = np.array(gt_va.columns)

    pout = {}
    pout['alpha'] = np.zeros(nsamp)
    pout['gamma'] = np.zeros(nsamp)
    pout['R0'] = np.zeros(nsamp)
    pout['beta'] = np.zeros(nsamp)

    pout['alpha'][0] = 0.2
    pout['gamma'][0] = 0.2
    pout['R0'][0] = 2.5
    pout['beta'][0] = 0.5

    params['beta'] = np.full((len(patch_df), params['T']), pout['beta'][0])
    params['alpha'] = pout['alpha'][0]
    params['gamma'] = pout['gamma'][0]

    #loglik_0 = pool.map(partial(get_LL, configs, patch_df, params, Theta, seeds, vaxs, gt_va, cov_mat_dict), random_seeds)
    loglik_0 = get_LL(configs, patch_df, params, Theta, seeds, vaxs, gt_va, cov_mat_dict,  gt_FIPS)
    for _iter in range(1, nsamp):

        ## update gamma

        #gamma_new = np.random.uniform(pout['gamma'][_iter-1]*0.9, pout['gamma'][_iter-1]*1.1)
        gamma_new = np.random.uniform(0.2, 0.4)
        params['gamma'] = gamma_new

        #loglik_1 = pool.map(partial(get_LL, configs, patch_df, params, Theta, seeds, vaxs, gt_va, cov_mat_dict), random_seeds)
        #acc_prob = np.mean(np.array(loglik_1) - np.array(loglik_0))

        loglik_1 = get_LL(configs, patch_df, params, Theta, seeds, vaxs, gt_va, cov_mat_dict,  gt_FIPS)
        acc_prob = loglik_1 - loglik_0

        if (np.random.uniform() < np.exp(acc_prob)):
            pout['gamma'][_iter] = gamma_new
            loglik_0 = loglik_1
        else:
            pout['gamma'][_iter] = pout['gamma'][_iter - 1]

        params['gamma'] = pout['gamma'][_iter]

        ## update beta through R0

        #R0_new = np.random.uniform(pout['R0'][_iter-1]*0.9, pout['R0'][_iter-1]*1.1)
        R0_new = np.random.uniform(2.1,2.8)
        beta_new = R0_new*gamma_new
        params['beta'] = np.full((len(patch_df), params['T']), beta_new)

        #loglik_1 = pool.map(partial(get_LL, configs, patch_df, params, Theta, seeds, vaxs, gt_va, cov_mat_dict), random_seeds)
        #acc_prob = np.mean(np.array(loglik_1) - np.array(loglik_0))

        # Replacing the direct objective call...
        # loglik_1 = get_LL(configs, patch_df, params, Theta, seeds, vaxs, gt_va, cov_mat_dict,  gt_FIPS)
        # ... with a request to the libensemble manager
        # Receive values from manager
        H0 = np.zeros(1, dtype=gen_specs['out'])
        H0[0]['alpha'] = params['alpha']
        H0[0]['beta'] = params['beta']
        H0[0]['gamma'] = params['gamma']
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H0)
        if tag in [STOP_TAG, PERSIS_STOP]:
            break

        acc_prob = loglik_1 - loglik_0

        if (np.random.uniform() < np.exp(acc_prob)):
            pout['beta'][_iter] = beta_new
            pout['R0'][_iter] = R0_new
            loglik_0 = loglik_1
        else:
            pout['beta'][_iter] = pout['beta'][_iter - 1]
            pout['R0'][_iter] = pout['R0'][_iter - 1]

        params['beta'] = np.full((len(patch_df), params['T']), pout['beta'][_iter])

        pout['alpha'][_iter] = pout['alpha'][0]
        params['alpha'] = pout['alpha'][_iter]

    return pout

def main(gt_datapath, state, patch_input_datadir, nsamp, gen_specs, libE_info):

    gt_va = get_state_gt(gt_datapath, state)
    configs, patch_df, params, seeds, Theta, vaxs = setupModel(patch_input_datadir)
    cov_mat_dict = get_cov(gt_va)

    pout = run_mcmc(configs, patch_df, params, seeds, Theta, vaxs, gt_va,
            cov_mat_dict, nsamp, gen_specs, libE_info)

    #print(pout)
    return pout

# gt_datapath = "us-counties.csv"
# state = "Virginia"
# patch_input_datadir = "/home/jlarson/research/libensemble/libensemble/tests/scaling_tests/arindam/patchsim/input/"
# nsamp = 10

# main(gt_datapath, state, patch_input_datadir, nsamp)

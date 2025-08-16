import pickle
import os
from math import comb

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy

import argparse
import time

def list_of_int(arg):
    return list(map(int, arg.split(',')))
def list_of_float(arg):
    return list(map(float, arg.split(',')))
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ts", type=list_of_int, default=None)
    parser.add_argument("-runs", type=list_of_int, default=None)
    parser.add_argument("-Ogammas", type=list_of_float, default=None)
    
    args = parser.parse_args()
    return args
args = get_arguments()
print(args)

fraction = 50000
L = 20
M = fraction * L
d = 2**L
init=1; add='_zerosz_p_one'
for g,gamma in enumerate(args.Ogammas):

    
    attn_avg = np.zeros((np.array(args.runs).max()+1, 4, 226)); attn_avg[:] = np.nan
    attn_std = np.zeros((np.array(args.runs).max()+1, 4, 226)); attn_std[:] = np.nan
    
    fn = f'Data_out/PhaseRecognition/g2_saved_models_MIPT/MIPT_QuAN4_Tbt2_L20_zerosz_one_g0.100vs0.850_f832_run_106/'+\
        f'attnlarge{gamma}_transfer_20_1_g0.1vs0.85_f40000-QuAN4_Tbt2_set64__h16nh1_ch4ker2st1'+\
        '_from_noiseless_20_1_g0.1vs0.85_f832-QuAN4_Tbt2_set64__h16nh1_ch4ker2st1.npz'
    data = np.load(fn)
    trajs = data['input']
    attn = data['sA0_wosm'].reshape(-1)
    powers_of_two = 2**np.arange(L-1, -1, -1, dtype=object)
    for t in args.ts:
        print(t)

        fn = f'Data_out/PhaseRecognition/counts_M={fraction},L={L},add={add},init={init},gamma={gamma},perm=t={t}.npz'
        prob = np.load(fn)['c']
        if prob[0] ==20: prob[0] = prob[0] // L
        if prob[-1] == 20: prob[-1] = prob[-1] // L
        print(prob.max())
        ptraj = np.zeros(len(trajs)).astype(int)
        for i,traj in enumerate(trajs):
            mt = traj[t-1].dot(powers_of_two)
            ptraj[i] = prob[mt]
            iters = 0
            while ptraj[i]==0:
                mt_ = mt + np.random.choice(np.arange(2**4)-2**3)
                ptraj[i] = prob[mt_]
                iters += 1
                if iters > 25: ptraj[i] = 1
        ptraj_outer = (ptraj.reshape(-1, 1) * ptraj.reshape(1, -1)).reshape(-1)
        print(gamma, t, ptraj_outer.max())
        run = args.runs[0]
        for i in tqdm(range(1,ptraj_outer.max()+1)):
            attn_avg[run, g, i] = attn[ptraj_outer==i].mean()
            if i==48: print(attn[ptraj_outer==i], attn_avg[run, g, i])
            attn_std[run, g, i] = scipy.stats.sem(attn[ptraj_outer==i])
        np.savez(f'Data_out/PhaseRecognition/interattn_t={t}_x_{run}_{gamma}', avg=attn_avg[run,g], std=attn_std[run,g])
        del ptraj_outer


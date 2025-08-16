import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import os
import argparse
import tqdm
import time
def list_of_int(arg):
    return list(map(int, arg.split(',')))
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", type=int, default=None)
    parser.add_argument("-M", type=int, default=50000)
    parser.add_argument("-init", type=int, default=1)
    parser.add_argument("-gamma", type=float, default=0.9)
    parser.add_argument("-task", type=str, default='_zerosz')
    args = parser.parse_args()
    return args

def gen_data_wkmsmt_1gamma(L, gamma, init, task, verbose=True):
    nt = 2*L*L
    dir_path = ''
    setsize = 64
    if task=='StateDistinguish': 
        path = dir_path+f'Data_src/{task}/L_{L}/wk_msmt_traj_{L}_'
        maxnfile = 200
    elif task=='PhaseRecognition': 
        path = dir_path+f'Data_src/{task}/L_{L}/wk_msmt_traj_{L}_'
        maxnfile = 300
    
    data = np.load(path+f'g{gamma:.3f}_s{init}_0.npz')
    a = data['s'].astype(bool)
    x1 = a.reshape(a.shape[0], nt)
    for circi in range(1, maxnfile):
        fpath = path+f'g{gamma:.3f}_s{init}_{circi}.npz'
        if not os.path.isfile(fpath): continue
        data = np.load(fpath)
        a = data['s'].astype(bool)
        xx = a.reshape(a.shape[0], nt)
        x1 = np.concatenate([x1,xx], axis = 0)
        del a, xx
    x1 = x1.reshape(-1, nt)[::-1]
    if verbose:
        print(f'gen_data_wkmsmt_1gamma from {path}g{gamma:.3f}_s{init}_0.npz:', f'shapes: {x1.shape}')
    return x1.astype(int)

def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))


args = get_arguments()
task = args.task
L = args.L
M = args.M
gamma = args.gamma
init = args.init


permutations = np.array([np.roll(np.arange(2*L), shift=-s) for s in range(2*L)])
permutations_name = np.array([f't={t+1}' for t in range(2*L)])
powers_of_two = 2**np.arange(2*L*L-1, -1, -1, dtype=object)
print(powers_of_two[0])
print(permutations_name)

start = time.time()
if gamma==0: barr = np.random.randint(0,2,(M, 2*L*L)).astype(bool)
else: barr = gen_data_wkmsmt_1gamma(L, gamma, init, task).astype(bool).reshape(-1, 2*L, L)
c_barr = np.array([np.roll(barr.reshape(-1, 2*L, L)[:M], shift=s, axis=-1) for s in tqdm.tqdm(range(L))]).reshape(-1, 2*L, L)
print('time(min)=', (time.time()-start)/60)
for p,perm in tqdm.tqdm(enumerate(permutations)):
    start = time.time()
    canonical_ints = c_barr[:,perm].reshape(-1, 2*L*L).dot(powers_of_two)
    
    print(len(canonical_ints))
    counts, bin_edges = np.histogram(np.array(canonical_ints, dtype=float), bins=np.linspace(0, 2**(2*L*L), 2**L+1))
    print(counts.max()/len(canonical_ints))
    np.savez(f'Data_out/{task}/counts_M={M},L={L},init={init},gamma={gamma},perm={permutations_name[p]}', c=counts)
    print('time(min)=', (time.time()-start)/60)
    
del barr, c_barr




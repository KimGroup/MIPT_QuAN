import numpy as np
import math
import time
from scipy.stats import sem
import scipy as sc

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import pickle


import wandb # type: ignore

import argparse
from tqdm import tqdm

def rolling(x):
    xx = x.copy()
    xx[:,:,1::2] = np.roll(x[:,:,1::2], 1, axis=-1)
    return xx
def gen_data_wkmsmt_1gamma(args, L, gamma, n_bitstring, verbose=True, maxnfile = 50):
    nt = 2*L*L
    dir_path = args.prefix #''
    path = dir_path+f'/MIPT_QuAN/Data_out/PhaseRecognition/tcorr_zerosz_L{L}.npz'
    
    data = np.load(path+f'g{gamma:.3f}_s{args.init}_0.npz')
    a = data['s'].astype(bool)
    x1 = a.reshape(a.shape[0], nt)
    for circi in range(1, maxnfile):
        data = np.load(path+f'g{gamma:.3f}_s{args.init}_{circi}.npz')
        a = data['s'].astype(bool)
        xx = a.reshape(a.shape[0], nt)
        x1 = np.concatenate([x1,xx], axis = 0)
        del a, xx
    x1 = x1.reshape(1, -1, nt)
    x1 = x1[:, :n_bitstring]
    if verbose:
        print(f'gen_data_wkmsmt_1gamma from {path}: {gamma:.3f}', f'shapes: {x1.shape}')
    return x1


class testing_set(Dataset):
    def __init__(self, xt, yt):
        xx = torch.from_numpy(xt)
        yy = torch.from_numpy(yt)
        self.X = xx
        self.Y = yy
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

class testing_set2(Dataset):
    def __init__(self, xt):
        xx = torch.from_numpy(xt)
        self.X = xx
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx]]

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
def list_of_float(arg):
    return list(map(float, arg.split(',')))
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_mini", help = "# of mini set", type = int, default = 1)
    parser.add_argument("-L", help = "L", type = int, default = None)
    parser.add_argument("-init", help = "0 or +", type = int, default = 1)
    parser.add_argument("-gamma1", type = float, default = 0.2)
    parser.add_argument("-gamma2", type = float, default = 0.9)
    parser.add_argument("-noise", help = "noise", type = str, default = '')

    parser.add_argument("-prefix", help = "prefix", type = str, default = None)
    parser.add_argument("-saveprefix", help = "save folder prefix", type = str, default = 'g2_saved_models_MIPT')
    args = parser.parse_args()

    return args
def is_non_zero_file(fpath, verb=False):
    result = os.path.isfile(fpath) and os.path.getsize(fpath) > 0
    if verb: print(f'exist={result}', fpath)
    return result

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

def my_function(optimizer):
    for g in optimizer.param_groups:
        g['lr'] *= 0.65
    return optimizer

def Cor(mat, xy, T, L, block=False):
    x, y = xy[0], xy[1]
    mat = mat.reshape(-1, T, L).astype(float)
    if block: mat = (mat+1)%4<2
    mat = mat*2 - 1
    
    cor_mat = np.zeros_like(mat)
    for k in range(mat.shape[0]):
        for i in range(mat.shape[1]):
            for j in range(mat.shape[2]):
                cor_mat[k, i,j] = mat[k, x, y] * mat[k, i, j]
    return cor_mat #.mean(axis=0)

def Con_cor(mat, xy, T, L, block=False):
    x, y = xy[0], xy[1]
    cor_mat = Cor(mat, xy, T, L, block=block)
    mat = mat.reshape(-1, T, L).astype(float)
    if block: mat = (mat+1)%4<2
    mat = mat*2 - 1
    return cor_mat - mat.mean(axis=0)*mat[:, x,y].mean()

def main():

    args = get_arguments()

    dir_add = f'_zerosz_p_{args.protocol}{args.noise}' # '_random' #
    gammas = np.array([0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, \
                        0.3, 0.325, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, \
                        0.75, 0.8, 0.85, 0.9])
    
    gammas_attn = gammas
    print(gammas, gammas_attn)

    gammas1 = gammas[gammas<=args.gamma1]
    gammas2 = gammas[gammas>=args.gamma2]
    gammas_t = np.setdiff1d(np.setdiff1d(gammas, gammas1),gammas2)

    setsizemodel = args.setsize
    fraction = int(args.fraction)
    n_bitstring = fraction // args.setsize * args.setsize
    if n_bitstring <= 50000: 
        basic_size = np.min([n_bitstring, 40000 // args.setsize * args.setsize // 4])
    else:
        sys.exit("ERROR: larger than total sample size")
    print('n_bitstring, basic_size, args.setsize, setsizemodel=', n_bitstring, basic_size, args.setsize, setsizemodel)

    Nt = args.L**2
    L = args.L
    datatype=float
    arr_Cor = np.zeros((len(gammas_attn), 4, 2*L, L, 2*L, L))
    xs = np.array([int(np.median(np.arange(L))-0.5), int(np.median(np.arange(L))+0.5)])
    x = xs[xs%2==1][0]  
    tcorrfilename = f'/MIPT_QuAN/Data_out/PhaseRecognition/tcorr_zerosz_L{L}.npz'
    
    for g,gamma in enumerate(gammas_attn):
        x5 = gen_data_wkmsmt_1gamma(args, args.L, gamma, maxnfile=50, n_bitstring=np.min([basic_size, 50000]), add=dir_add).astype(datatype)
            
        x5 = x5.reshape(-1, 2*args.L, args.L)
        for t0 in tqdm(range(2*L)):
            for x0 in range(L):
                mid = Cor(x5, (t0,x0), 2*L, L)
                for m in range(4):
                    if m==0: arr_Cor[g,m,t0,x0] = np.mean(mid, axis=0)
                    else: arr_Cor[g,m,t0,x0] = sc.stats.moment(mid, axis=0, moment=m+1)
        print(x5.shape, arr_Cor.shape)
    np.savez(tcorrfilename, arrC=arr_Cor)




if __name__ == "__main__":
    main()

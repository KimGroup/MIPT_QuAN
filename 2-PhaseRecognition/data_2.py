import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import math
import sys
import os
import pickle

def gen_data_wkmsmt_1gamma(args, L, gamma, n_bitstring, verbose=True, maxnfile = 150):
    nt = 2*L*L
    dir_path = args.prefix #''
    path = dir_path+f'Data_src/PhaseRecognition/L_{L}{args.noise}/wk_msmt_traj_{L}_'
    
    data = np.load(path+f'g{gamma:.3f}_s{args.init}_0.npz')
    a = data['s'].astype(bool)
    x1 = a.reshape(a.shape[0], nt)
    for circi in range(1, maxnfile):
        fpath = path+f'g{gamma:.3f}_s{args.init}_{circi}.npz'
        if not os.path.isfile(fpath): continue
        data = np.load(fpath)
        a = data['s'].astype(bool)
        xx = a.reshape(a.shape[0], nt)
        x1 = np.concatenate([x1,xx], axis = 0)
        del a, xx
    x1 = x1.reshape(-1, nt)
    n_truncate = x1.shape[0] - x1.shape[0]%args.setsize
    x1 = x1[:n_truncate] 
    p = np.random.permutation(x1.shape[0]); x1 = x1[p]; x1 = x1[:n_bitstring]
    if verbose:
        print(f'gen_data_wkmsmt_1gamma from {path}g{gamma:.3f}_s{args.init}_0.npz:', f'shapes: {x1.shape}')
    return x1
def gen_data_wkmsmt(args, gammas, L, n_bitstring, verbose=True, basic_size=5000, maxnfile=150):
    nt = 2*L*L
    dir_path = args.prefix #''
    path = dir_path+f'Data_src/PhaseRecognition/L_{L}{args.noise}/wk_msmt_traj_{L}_'
    gammas1 = gammas[gammas<=args.gamma1]
    gammas2 = gammas[gammas>=args.gamma2]

    # class-0 for training
    x1_ = []
    x3t_ = []
    for g1 in gammas1:
        data = np.load(path+f'g{g1:.3f}_s{args.init}_0.npz')
        a = data['s'].astype(bool)
        x1 = a.reshape(a.shape[0], nt)
        for circi in range(1,maxnfile):
            fpath = path+f'g{g1:.3f}_s{args.init}_{circi}.npz'
            if not os.path.isfile(fpath): continue
            data = np.load(fpath)
            a = data['s'].astype(bool)
            xx = a.reshape(a.shape[0], nt)
            x1 = np.concatenate([x1,xx], axis = 0)
            del a, xx
        p = np.random.permutation(x1.shape[0]); x1 = x1[p]
        x1_.append(x1[:int(n_bitstring)])
        x3t_.append(x1[-int(basic_size):])
    
    # class-1 for training
    x2_ = []
    x4t_ = []
    for g2 in gammas2:
        data = np.load(path+f'g{g2:.3f}_s{args.init}_0.npz')
        a = data['s'].astype(bool)
        x2 = a.reshape(a.shape[0], nt)
        for circi in range(1,maxnfile):
            fpath = path+f'g{g2:.3f}_s{args.init}_{circi}.npz'
            if not os.path.isfile(fpath): continue
            data = np.load(fpath)
            a = data['s'].astype(bool)
            xx = a.reshape(a.shape[0], nt)
            x2 = np.concatenate([x2,xx], axis = 0)
            del a, xx
        p = np.random.permutation(x2.shape[0]); x2 = x2[p]
        x2_.append(x2[:int(n_bitstring)])
        x4t_.append(x2[-int(basic_size):])
    
    x1_ = np.array(x1_)
    x3t_ = np.array(x3t_)
    x2_ = np.array(x2_)
    x4t_ = np.array(x4t_)

    if verbose:
        print(f'gen_data_wkmsmt from {path}: {args.gamma1:.2f}vs{args.gamma2:.2f}')
        print(f'shapes: {x1_.shape}, {x2_.shape}, {x3t_.shape}, {x4t_.shape}')
    return x1_, x2_, x3t_, x4t_


def rolling(x):
    xx = x.copy()
    xx[:,:,1::2] = np.roll(x[:,:,1::2], 1, axis=-1)
    return xx
def gen_data_wkmsmt_1gamma_embedding(args, L, gamma, n_bitstring, maxnfile = 50, verb=True):
    _x1 = gen_data_wkmsmt_1gamma(args, L, gamma, n_bitstring, verbose=verb, maxnfile = maxnfile)
    _x1 = _x1.astype(int)
    _x1 = _x1.reshape(-1, 2*args.L*args.L)
    x1 = np.dot(rolling(_x1.reshape(-1, n_bitstring, args.L*2, args.L)).reshape(-1, n_bitstring, args.L*args.L, 2), np.array([2,1]))
    x1 = x1.reshape(-1, args.setsize, args.L*args.L)
    if verb: print(f'gen_data_wkmsmt_1gamma_embedding: {gamma:.3f}', f'shapes: {x1.shape}')
    return x1
def gen_data_wkmsmt_embedding(args, gammas, L, n_bitstring, basic_size=5000, maxnfile=150):
    _x1, _x2, _x3t, _x4t = gen_data_wkmsmt(args, gammas, L, n_bitstring, verbose=True, basic_size=basic_size, maxnfile=maxnfile)
    _x1, _x2, _x3t, _x4t = _x1.astype(int), _x2.astype(int), _x3t.astype(int), _x4t.astype(int)
    model_name = f'{args.L}_{args.init}_g{args.gamma1}vs{args.gamma2}_f{args.fraction}'\
                +f'-{args.modelnum}_set{args.setsize}__h{args.hdim}nh{args.nhead}_ch{args.ch}ker2st1' 
    np.savez_compressed(f'{args.saveprefix}/{args.wandb_name}/useddata_{model_name}', x1=_x1, x2=_x2)
    x1 = np.dot(rolling(_x1.reshape(-1, n_bitstring, args.L*2, args.L)).reshape(-1, n_bitstring, args.L*args.L, 2), np.array([2,1]))
    x2 = np.dot(rolling(_x2.reshape(-1, n_bitstring, args.L*2, args.L)).reshape(-1, n_bitstring, args.L*args.L, 2), np.array([2,1]))
    x3t = np.dot(rolling(_x3t.reshape(-1, basic_size, args.L*2, args.L)).reshape(-1, basic_size, args.L*args.L, 2), np.array([2,1]))
    x4t = np.dot(rolling(_x4t.reshape(-1, basic_size, args.L*2, args.L)).reshape(-1, basic_size, args.L*args.L, 2), np.array([2,1]))
    print(f'gen_data_wkmsmt_embedding: {args.gamma1:.2f}vs{args.gamma2:.2f}')
    print(f'shapes: {x1.shape}, {x3t.shape}')
    return x1, x2, x3t, x4t

def gen_data_sample(set_size, x1, x2, Nt, shuffle=True, datatype=bool, verb=True):
    if x1.reshape(-1, Nt).shape[0] % set_size == 0:
        xx1 = x1.reshape(-1, set_size, Nt)
        xx2 = x2.reshape(-1, set_size, Nt)
    else: 
        x1shape = x1.reshape(-1, Nt).shape[0]
        n_truncate = x1shape - x1shape%set_size
        xx1 = x1.reshape(-1, Nt)[:n_truncate].reshape(-1, set_size, Nt)
        xx2 = x2.reshape(-1, Nt)[:n_truncate].reshape(-1, set_size, Nt)
    if verb: print('gen_data_sample', x1.shape, xx1.shape)
    y1 = np.zeros(xx1.shape[0], dtype=datatype)
    y2 = np.ones(xx2.shape[0], dtype=datatype)

    x = np.concatenate([xx1, xx2], axis = 0).astype(datatype)
    del xx1, xx2
    y = np.concatenate([y1, y2], axis = 0)
    y = np.expand_dims(y, axis=1)

    if shuffle:
        p = np.random.permutation(len(x)) # mix set.
        x, y = x[p], y[p]
    return x, y

class training_set(Dataset):
    def __init__(self, x, y):
        xx = torch.from_numpy(x)
        yy = torch.from_numpy(y)
        self.X = xx
        self.Y = yy
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

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
    


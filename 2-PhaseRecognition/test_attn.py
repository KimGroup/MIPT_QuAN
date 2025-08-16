import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import math
import time
from scipy.stats import sem
import scipy as sc

from collections import OrderedDict
import sys
import os
import pickle

from models_quan import *

import wandb # type: ignore

import argparse
from tqdm import tqdm

def rolling(x):
    xx = x.copy()
    xx[:,:,1::2] = np.roll(x[:,:,1::2], 1, axis=-1)
    return xx
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
    x1 = x1.reshape(1, -1, nt)
    x1 = x1[:, :n_bitstring]
    if verbose:
        print(f'gen_data_wkmsmt_1gamma from {path}: {gamma:.3f}', f'shapes: {x1.shape}')
    return x1

def gen_data_wkmsmt_1gamma_embedding(args, L, gamma, n_bitstring, maxnfile = 150, verb=True):
    _x1 = gen_data_wkmsmt_1gamma(args, L, gamma, n_bitstring, verbose=verb, maxnfile = maxnfile)
    _x1 = _x1.astype(int)
    n_bitstring = _x1.shape[1]
    x1 = np.dot(rolling(_x1.reshape(-1, n_bitstring, args.L*2, args.L)).reshape(-1, n_bitstring, args.L*args.L, 2), np.array([2,1]))
    x1 = x1.reshape(-1, args.L*args.L)
    n_truncate = x1.shape[0] - x1.shape[0]%args.setsize
    x1 = x1.reshape(-1, args.L*args.L)[:n_truncate].reshape(-1, args.setsize, args.L*args.L)
    if verb: print(f'gen_data_wkmsmt_1gamma_embedding: {gamma:.3f}', f'shapes: {x1.shape}')
    return _x1, x1

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
def list_of_int(arg):
        return list(map(int, arg.split(',')))
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-modelprevname", type=str, default='e')
    parser.add_argument("-n_mini", help = "# of mini set", type = int, default = 1)
    parser.add_argument("-L", help = "L", type = int, default = None)
    parser.add_argument("-init", help = "0 or +", type = int, default = 0)
    parser.add_argument("-gamma1", type = float, default = 0.2)
    parser.add_argument("-gamma2", type = float, default = 0.9)
    parser.add_argument("-fraction", type = str, default = '1')
    parser.add_argument("-fractionmodel", type = str, default = '1')
    parser.add_argument("-setsize", help = "setsize", type = int, default = None)
    parser.add_argument("-noise", help = "noise", type = str, default = '')

    parser.add_argument("-prefix", help = "prefix", type = str, default = None)
    parser.add_argument("-saveprefix", help = "save folder prefix", type = str, default = 'g2_saved_models_MIPT')
    parser.add_argument("-batchsize", type=int, default=50) 

    parser.add_argument("-modelnum", type = str, default = None)
    parser.add_argument("-hdim", type = int, default = 64)
    parser.add_argument("-nhead", type = int, default = 16)

    parser.add_argument("-ch", type = int, default = 32)
    parser.add_argument("-dim_outputs", type=int, default=1)
    parser.add_argument("-p_outputs", type=int, default=1)

    parser.add_argument("-drop", type = float, default = 0)
    parser.add_argument("-lr", type = float, default = 1e-4)
    parser.add_argument("-prev", type = str, default = None)

    parser.add_argument("-wandb_name", type = str)
    parser.add_argument("-debug", action='store_true', default=False)
    parser.add_argument("-test", action='store_true',default=False)
    
    parser.add_argument("-shuffle_epoch", type=int, default=10, help='shuffling dataset every n epoch')

    
    parser.add_argument("-ts", type=list_of_int, default=None)
    parser.add_argument("-runs", type=list_of_int, default=None)
    
    args = parser.parse_args()
    return args
def is_non_zero_file(fpath, verb=False):
    result = os.path.isfile(fpath) and os.path.getsize(fpath) > 0
    if verb: print(f'exist={result}', fpath)
    return result


def Cor(mat, xy, T, L, block=False):
    x, y = xy[0], xy[1]
    mat = mat.reshape(-1, T, L).astype(float)
    if block: mat = (mat+1)%4<2
    mat = mat*2 - 1
    
    cor_mat = np.zeros_like(mat)
    for k in range(mat.shape[0]):
        cor_mat[k] = mat[k, x, y] * mat[k]
    return cor_mat.mean(axis=0)

def main():

    args = get_arguments()
    gammas = np.array([0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.21, 0.225, 0.25, 0.275, \
                        0.3, 0.325, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, \
                        0.7, 0.75, 0.8, 0.85, 0.9,  0.95])
    
    gammas_attn = [0.05, 0.9]
    print(gammas, gammas_attn)

    gammas1 = gammas[gammas<=args.gamma1]
    gammas2 = gammas[gammas>=args.gamma2]
    gammas_t = np.setdiff1d(np.setdiff1d(gammas, gammas1),gammas2)

    Nt = args.L**2
    datatype=int
    os.makedirs(f'{args.saveprefix}',exist_ok=True)
    os.makedirs(f'{args.saveprefix}/{args.wandb_name}',exist_ok=True)


    setsizemodel = args.setsize
    fraction = int(args.fraction)
    n_bitstring = fraction // args.setsize * args.setsize
    basic_size = 50000 // args.setsize * args.setsize

    print('n_bitstring, basic_size, args.setsize, setsizemodel=', n_bitstring, basic_size, args.setsize, setsizemodel)
    data_name = f'{args.L}_{args.init}_g{args.gamma1}vs{args.gamma2}_f{args.fraction}'\
                +f'-{args.modelnum}_set{args.setsize}__h{args.hdim}nh{args.nhead}_ch{args.ch}ker2st1' 
    model_name = f'{args.L}_{args.init}_g{args.gamma1}vs{args.gamma2}_f{args.fractionmodel}'\
                +f'-{args.modelnum}_set{setsizemodel}__h{args.hdim}nh{args.nhead}_ch{args.ch}ker2st1' 
    model_prev_dir = f'{args.saveprefix}/'+args.wandb_name 
    model_prev_name = f'{model_prev_dir}/model_vl_{model_name}.pth'
        
    attnfilename = f'{args.saveprefix}/{args.wandb_name}/attn_transfer_{data_name}_from_noiseless_{model_name}.npz'
    print('Transfer learning: f_out = ', attnfilename)
    print('Transfer learning: model prev = ', model_prev_name)

    if args.test==True:
        test_loaders = []
        test_input_loaders = []
        for gamma in gammas_attn:
            _x5, x5 = gen_data_wkmsmt_1gamma_embedding(args, args.L, gamma, n_bitstring=np.min([basic_size, 50000]), maxnfile=200)
            test_input_loaders.append(_x5.reshape(-1,2*args.L, args.L).astype(datatype))
            test_loaders.append(DataLoader(dataset = testing_set2(x5.astype(datatype)), batch_size=args.batchsize * 10, shuffle=False))
    
        print('test loader loaded, len:', len(test_loaders))
    
    
    start = time.time() 
    if 'QuAN8_Tbt' in args.modelnum: 
        model = QuAN_Tbt(setsize=args.setsize, channel=args.ch, dim_output=args.dim_outputs, \
                                    dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.L*2, Nc=args.L//2, \
                                    p_outputs=args.p_outputs, sab_L=int(args.modelnum[-1]), sab_M=3, dropout=args.drop)
    elif 'QuAN4_Tbt' in args.modelnum: 
        model = QuAN_Tbt(setsize=args.setsize, channel=args.ch, dim_output=args.dim_outputs, \
                                    dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.L*2, Nc=args.L//2, \
                                    p_outputs=args.p_outputs, sab_L=int(args.modelnum[-1]), sab_M=2, dropout=args.drop)
    elif 'QuAN2_Tbt' in args.modelnum: 
        model = QuAN_Tbt(setsize=args.setsize, channel=args.ch, dim_output=args.dim_outputs, \
                                    dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.L*2, Nc=args.L//2, \
                                    p_outputs=args.p_outputs, sab_L=int(args.modelnum[-1]), sab_M=1, dropout=args.drop)
    elif 'SMLP_Tbm' in args.modelnum: 
        model = SMLP_Tbm(setsize=args.setsize, channel=args.ch, dim_output=args.dim_outputs, \
                                    dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.L*2, Nc=args.L//2, \
                                    p_outputs=args.p_outputs, sab_L=int(args.modelnum[-1]), sab_M=1, dropout=args.drop)
    elif 'SMLP_Tbt' in args.modelnum: 
        model = SMLP_Tbt(setsize=args.setsize, channel=args.ch, dim_output=args.dim_outputs, \
                                    dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.L*2, Nc=args.L//2, \
                                    p_outputs=args.p_outputs, sab_L=int(args.modelnum[-1]), sab_M=1, dropout=args.drop)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'total parameters: {total_params}') 
    model = torch.nn.DataParallel(model)

    if args.wandb_name and not args.debug:
        wandb.init(project='MIPT', name=args.wandb_name)
        wandb.log({'# of parameters': total_params})

    print("Previous model:", model_prev_name)
    state_dict = torch.load(model_prev_name)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('module.module', 'module')
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    print('model loaded.')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device); model.to(device)
    
    
    
    # validation
    model.eval()

    if args.test==True:
        attns = []
        for j,test_loader in enumerate(test_loaders):
            attnfilename = f'{args.saveprefix}/{args.wandb_name}/attnlarge{gammas_attn[j]}_transfer_{data_name}_from_noiseless_{model_name}.npz'
            
            mat_output = []
            mat_mid_output = []
            for _,testdata in enumerate(test_loader):
                test_inputs = testdata[0].to(device)
                test_outputs = model.module(test_inputs)
                test_mid_outputs = model.module.aftertemp.detach().cpu().numpy().reshape(-1, args.L)
                mat_mid_output.extend(test_mid_outputs)
                mat_output.extend(test_outputs.detach().cpu().numpy())
                print(gammas_attn[j], test_inputs.detach().cpu().numpy().shape, test_outputs.detach().cpu().numpy().shape, \
                        model.module.transformer[0].mab.A.detach().cpu().numpy().shape, \
                        model.module.enc[0].mab.A.detach().cpu().numpy().shape, \
                        model.module.dec[0].mab.A.detach().cpu().numpy().shape)
            _x5 = test_input_loaders[j]
            print(_x5.shape)

            midoutput = np.array(mat_mid_output)

            sA0_wosm = np.zeros((len(midoutput), len(midoutput)))

            Q = model.module.enc[0].mab.fc_q(torch.from_numpy(midoutput).to(device)).detach().cpu().numpy().reshape(-1, args.L)
            K = model.module.enc[0].mab.fc_k(torch.from_numpy(midoutput).to(device)).detach().cpu().numpy().reshape(-1, args.L)
            print(model.module.enc[0].mab.fc_k.weight.detach().cpu().numpy())
            
            for i in tqdm(range(len(midoutput))):
                sA0_wosm[i] = np.dot(Q[i], K.T) /np.sqrt(args.L)
            np.savez_compressed(attnfilename, sA0_wosm=sA0_wosm, input = _x5)
            print('saved as:', attnfilename)


    torch.cuda.empty_cache()
    print('Done: ', args.setsize, setsizemodel)
    print(f'Time: {time.time()-start} s, {(time.time()-start)/60} min, {(time.time()-start)/3600} hour \n\n')
    return attns


if __name__ == "__main__":
    main()

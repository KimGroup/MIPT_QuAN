import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import math
import time
import scipy as sc
from scipy.stats import sem

from collections import OrderedDict
import sys
import os
import pickle

from models_quan import *
from data_1 import *

import wandb 

import argparse
from tqdm import tqdm

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
    parser.add_argument("-n_mini", help = "# of mini set", type = int, default = 1)
    parser.add_argument("-L", help = "L", type = int, default = None)
    parser.add_argument("-init", help = "0 or +", type = int, default = 0)
    parser.add_argument("-gamma", type = float, default = 0.9)
    parser.add_argument("-runs", type=list_of_int, default=None)
    parser.add_argument("-fraction", type = str, default = '1')
    parser.add_argument("-setsize", help = "setsize", type = int, default = None)
    parser.add_argument("-noise", help = "noise", type = str, default = '')

    parser.add_argument("-prefix", help = "prefix", type = str, default = None)
    parser.add_argument("-saveprefix", help = "save folder prefix", type = str, default = 'g2_saved_models_MIPT')
    parser.add_argument("-epoch", help = "total epochs", type = int, default = 100)
    parser.add_argument("-batchsize", type=int, default=50) 

    parser.add_argument("-modelnum", type = str, default = None)
    parser.add_argument("-hdim", type = int, default = 64)
    parser.add_argument("-nhead", type = int, default = 16)

    parser.add_argument("-ch", type = int, default = 32)
    parser.add_argument("-dim_outputs", type=int, default=1)
    parser.add_argument("-p_outputs", type=int, default=1)

    parser.add_argument("-drop", type = float, default = 0)
    parser.add_argument("-lr", type = float, default = 1e-4)
    parser.add_argument("-lrn", type = int, default = 0)
    parser.add_argument("-prev", type = str, default = None)

    parser.add_argument("-wandb_name", type = str)
    parser.add_argument("-debug", action='store_true', default=False)
    parser.add_argument("-test", action='store_true',default=False)
    
    parser.add_argument("-shuffle_epoch", type=int, default=10, help='shuffling dataset every n epoch')
    args = parser.parse_args()

    return args
def is_non_zero_file(fpath, verb=False):
    result = os.path.isfile(fpath) and os.path.getsize(fpath) > 0
    if verb: print(f'exist={result}', fpath)
    return result

def main():

    args = get_arguments()
    Nt = args.L**2
    datatype=int
    
    fraction = int(args.fraction)
    n_bitstring = fraction // args.setsize * args.setsize
    if fraction>=3500000: basic_size = 1000000 // args.setsize * args.setsize
    else: basic_size = 200000 // args.setsize * args.setsize
    if n_bitstring+basic_size >= 7498360: sys.exit("ERROR: larger than total sample size")
    
    x1, x2, x3v, x4v = gen_data_wkmsmt_State_embedding(args, args.gamma, args.L, n_bitstring=n_bitstring, basic_size=basic_size, maxnfile = 200, add=f'')
    x1, x2, x3v, x4v = x1.astype(datatype), x2.astype(datatype), x3v.astype(datatype), x4v.astype(datatype)
    x5s = []
    xt, yt = gen_data_sample(args.setsize, x3v, x4v, Nt, datatype=datatype)
    val_loader = DataLoader(dataset = testing_set(xt, yt), batch_size=args.batchsize * 10, shuffle=True)
    del xt, yt
    
    for run in args.runs:
        print(run, 'out of', args.runs)
        model_name = f'{args.L}_g{args.gamma}_0vs{args.fraction}'\
                    +f'-{args.modelnum}_set{args.setsize}__h{args.hdim}nh{args.nhead}_ch{args.ch}ker2st1' 
        print(n_bitstring, basic_size, args.setsize)
        os.makedirs(f'{args.saveprefix}',exist_ok=True)
        os.makedirs(f'{args.saveprefix}/{args.wandb_name}_run_{run}',exist_ok=True)
            
        f = open(f'{args.saveprefix}/{args.wandb_name}_run_{run}/cred_{model_name}.txt', 'w')
        f.close()
        f = open(f'{args.saveprefix}/{args.wandb_name}_run_{run}/cred_{model_name}.txt', 'a')
        print('saved as: ', f'{args.saveprefix}/{args.wandb_name}_run_{run}/cred_{model_name}.txt')

        print("Storing as:", model_name)
        if 'QuAN32_Tbt' in args.modelnum: 
            model = QuANm_Tbt(setsize=args.setsize, channel=args.ch, dim_output=args.dim_outputs, \
                                        dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.L*2, Nc=args.L//2, \
                                        p_outputs=args.p_outputs, sab_L=int(args.modelnum[-1]), sab_M=4, dropout=args.drop)
        elif 'QuAN8_Tbt' in args.modelnum: 
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
        
        if args.prev != None:
            if args.prev == 'yes': 
                fn = f'{args.saveprefix}/{args.wandb_name}_run_{run}/log_{model_name}.txt'
                _,vl_ls,_,_,vl_ac,_,vl_ac0,_,vl_ac1,_ = np.loadtxt(fn, unpack=True)
                epoch = np.max([np.argmax(vl_ac), 25])
                args.prev = f'{args.saveprefix}/{args.wandb_name}_run_{run}/model_e{epoch}_{model_name}.pth'
                if not os.path.isfile(args.prev):
                    args.prev = f'{args.saveprefix}/{args.wandb_name}_run_{run}/model_vl_{model_name}.pth'
            else: continue
            print("Previous model:", args.prev)
            state_dict = torch.load(args.prev)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('module.module', 'module')
                new_state_dict[k]=v
            model.load_state_dict(new_state_dict)
        print('model loaded.:', args.prev)

        criterion = nn.BCELoss() 
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device); model.to(device)
        
        
        model.eval()
        mat_output0, mat_output1 = [], []
        val_acc = 0
        val_cred = 0
        for _,valdata in enumerate(val_loader):
            val_inputs, val_labels = valdata[0].to(device), valdata[1].to(device)
            val_outputs = model.module(val_inputs)
            
            del val_inputs
            if torch.isnan(val_outputs).any():
                print('NaN:', torch.isnan(val_outputs).sum())
            loss_ = criterion(val_outputs.squeeze(), val_labels.float().squeeze())

            m0 = val_labels.squeeze() < 0.5
            m1 = val_labels.squeeze() >= 0.5
            loss_.detach().cpu().numpy()
            mat_output0.extend(1-val_outputs[m0].detach().cpu().numpy())
            mat_output1.extend(val_outputs[m1].detach().cpu().numpy())
            
            val_pred_classes = torch.cat([1-val_outputs, val_outputs], axis=1).argmax(axis=1)
            val_acc += torch.count_nonzero(torch.eq(val_pred_classes, val_labels.squeeze())).item()/len(val_labels)
        val_acc /= len(val_loader)
        print(len(mat_output0+mat_output1))
        val_cred = np.array(mat_output0+mat_output1).mean()


        np.savetxt(f, np.array([val_acc,val_cred]), newline=" ")
        f.write("\n"); f.close()

        print(f'L={args.L}, set={args.setsize}, {args.gamma}, frac={args.fraction}')
        print(f'Te: {val_acc:.4f}, {val_cred:.4f}%')

        args.prev = 'yes'


if __name__ == "__main__":
    main()

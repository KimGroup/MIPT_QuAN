import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from data_2 import *

import wandb # type: ignore

import argparse
from tqdm import tqdm

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
def list_of_float(arg):
    return list(map(float, arg.split(',')))
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-modelprevname", type=str, default='vl')
    parser.add_argument("-n_mini", help = "# of mini set", type = int, default = 1)
    parser.add_argument("-L", help = "L", type = int, default = None)
    parser.add_argument("-init", help = "0 or +", type = int, default = 0)
    parser.add_argument("-gamma1", type = float, default = 0.2)
    parser.add_argument("-gamma2", type = float, default = 0.9)
    parser.add_argument("-fraction", type = str, default = '1')
    parser.add_argument("-fractionmodel", type = str, default = '1')
    parser.add_argument("-setsize", help = "setsize", type = int, default = None)
    parser.add_argument("-noise", help = "noise", type = str, default = '')
    parser.add_argument("-modelnoise", type = str, default = None)
    parser.add_argument("-datanoise", type = str, default = None)

    parser.add_argument("-prefix", help = "prefix", type = str, default = '/home/hk684/')
    parser.add_argument("-saveprefix", help = "save folder prefix", type = str, default = 'g2_saved_models_MIPT')
    parser.add_argument("-epoch", help = "total epochs", type = int, default = 100)
    parser.add_argument("-batchsize", type=int, default=50) 

    parser.add_argument("-modelnum", type = str, default = None)
    parser.add_argument("-hdim", type = int, default = 64)
    parser.add_argument("-nhead", type = int, default = 16)
    parser.add_argument("-emb", action = 'store_true', default = False)

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
    args = parser.parse_args()

    return args
def is_non_zero_file(fpath, verb=False):
    result = os.path.isfile(fpath) and os.path.getsize(fpath) > 0
    if verb: print(f'exist={result}', fpath)
    return result

def main():

    args = get_arguments()
    gammas = np.array([0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.21, 0.225, 0.25, 0.275, \
                        0.3, 0.325, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, \
                        0.7, 0.75, 0.8, 0.85, 0.9])
    print(gammas)

    gammas1 = gammas[gammas<=args.gamma1]
    gammas2 = gammas[gammas>=args.gamma2]
    gammas_t = np.setdiff1d(np.setdiff1d(gammas, gammas1),gammas2)


    Nt = args.L**2
    datatype=int
    os.makedirs(f'{args.saveprefix}',exist_ok=True)
    os.makedirs(f'{args.saveprefix}/{args.wandb_name}',exist_ok=True)


    for setsizemodel in [1024, 512, 256, 128, 64, 16, 4, 1]:
        if setsizemodel != args.setsize: continue
        fraction = int(args.fraction)
        n_bitstring = fraction // args.setsize * args.setsize
        if fraction==40000: basic_size = 10000 // args.setsize * args.setsize
        else: basic_size = fraction
        print('n_bitstring, basic_size, args.setsize=', n_bitstring, basic_size, args.setsize)
        data_name = f'{args.L}_{args.init}_g{args.gamma1}vs{args.gamma2}_f{args.fraction}'\
                    +f'-{args.modelnum}_set{args.setsize}__h{args.hdim}nh{args.nhead}_ch{args.ch}ker{args.ker}st{args.st}' 
        model_name = f'{args.L}_{args.init}_g{args.gamma1}vs{args.gamma2}_f{args.fractionmodel}'\
                    +f'-{args.modelnum}_set{setsizemodel}__h{args.hdim}nh{args.nhead}_ch{args.ch}ker{args.ker}st{args.st}' 
        
        if args.modelnoise=='noiseless':
            model_prev_dir = f'{args.saveprefix}/'+args.wandb_name.replace('_zerosz_one_noise', '_zerosz_one').replace(f'f{args.fraction}', f'f{args.fractionmodel}')
        elif args.modelnoise=='noisy':
            model_prev_dir = f'{args.saveprefix}/'+args.wandb_name.replace('_zerosz_one', '_zerosz_one_noise').replace(f'f{args.fraction}', f'f{args.fractionmodel}')
        
        cut = 400
        tr_ls, vl_ls, ts_ls, _, vl_ac, _, _, _, _, _ = np.loadtxt(f'{model_prev_dir}/log_{model_name}.txt', unpack=True, max_rows=cut)
        model_prev_name = f'{model_prev_dir}/model_vl_{model_name}.pth'
            
        f_out = open(f'{args.saveprefix}/{args.wandb_name}/phase_out_test_{args.datanoise}_{data_name}_from_{args.modelnoise}_{model_name}.txt', 'w')
        f = open(f'{args.saveprefix}/{args.wandb_name}/log_test_{args.datanoise}_{data_name}_from_{args.modelnoise}_{model_name}.txt', 'w')
        print('Transfer learning: f_out = ', f'{args.saveprefix}/{args.wandb_name}/phase_out_test_m_{args.datanoise}_{data_name}_from_{args.modelnoise}_{model_name}.txt')
        print('Transfer learning: model prev = ', model_prev_name)

        x1, x2, x3v, x4v = gen_data_wkmsmt_embedding(args, gammas, args.L, n_bitstring=np.min([basic_size, 50000]), basic_size=basic_size)
        x1, x2, x3v, x4v = x1.astype(datatype), x2.astype(datatype), x3v.astype(datatype), x4v.astype(datatype)
        
        xt, yt = gen_data_sample(args.setsize, x3v, x4v, Nt, datatype=datatype)
        val_loader = DataLoader(dataset = testing_set(xt, yt), batch_size=args.batchsize * 10, shuffle=True)
        del xt, yt
        if args.test==True:
            x5s = []
            x5 = gen_data_wkmsmt_1gamma_embedding(args, args.L, gammas1[0], n_bitstring=np.min([basic_size, 50000])).astype(datatype)
            for gamma in gammas1:
                if gamma==gammas1[0]: continue
                xx=gen_data_wkmsmt_1gamma_embedding(args, args.L, gamma, n_bitstring=np.min([basic_size, 50000])).astype(datatype)
                x5 = np.concatenate([x5,xx],axis=1)
            x5s.append(x5)
            for gamma in gammas_t:
                x5 = gen_data_wkmsmt_1gamma_embedding(args, args.L, gamma, n_bitstring=np.min([basic_size, 50000])).astype(datatype)
                x5s.append(x5)
                del x5
            x5 = gen_data_wkmsmt_1gamma_embedding(args, args.L, gammas2[0], n_bitstring=np.min([basic_size, 50000])).astype(datatype)
            for gamma in gammas2:
                if gamma==gammas2[0]: continue
                xx=gen_data_wkmsmt_1gamma_embedding(args, args.L, gamma, n_bitstring=np.min([basic_size, 50000])).astype(datatype)
                x5 = np.concatenate([x5,xx],axis=1)
            x5s.append(x5)

            nrepeat = np.max([10000//basic_size, 1])
            test_loaders = []
            for x5 in x5s:
                x5 = x5.reshape(-1, args.L*args.L)
                x5_ = np.array([x5[np.random.permutation(x5.shape[0])] for i in range(nrepeat)])
                x5_ = x5_.reshape(-1, args.setsize, args.L*args.L)
                test_loaders.append(DataLoader(dataset = testing_set2(x5_), batch_size=args.batchsize * 10, shuffle=True))

            print('test loader loaded:', len(test_loaders)==len(gammas_t)+2)
        
        
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
        
        criterion = nn.BCELoss()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device); model.to(device)
        
        
        running_loss, val_loss, test_loss = 0.0, 0.0, 0.0
        running_acc, val_acc, test_acc= 0.0, 0.0, 0.0
        val_acc0, val_acc1 = np.zeros(19), np.zeros(19)
        test_acc0, test_acc1 = 0.0, 0.0
        
        # validation
        model.eval()
        for _,valdata in enumerate(val_loader):
            val_inputs, val_labels = valdata[0].to(device), valdata[1].to(device)
            val_outputs = model.module(val_inputs)
            del val_inputs
            if torch.isnan(val_outputs).any():
                print('NaN:', torch.isnan(val_outputs).sum())
            loss_ = criterion(val_outputs.squeeze(), val_labels.float().squeeze())
            loss_.detach().cpu().numpy()
            
            val_pred_classes = torch.cat([1-val_outputs, val_outputs], axis=1).argmax(axis=1)
            val_loss += loss_.item()
            val_acc += torch.count_nonzero(torch.eq(val_pred_classes, val_labels.squeeze())).item()/len(val_labels)
            del loss_, val_outputs
            for i_th, threshold in enumerate(np.arange(1,20)*0.05):
                m0 = val_labels.squeeze() < threshold
                m1 = val_labels.squeeze() > threshold
                if len(val_labels.squeeze()[m0]) == 0: val_acc0[i_th] += 0.0
                else: val_acc0[i_th] += torch.count_nonzero(torch.eq(val_pred_classes[m0],val_labels.squeeze()[m0])).item()/len(val_labels.squeeze()[m0])
                if len(val_labels.squeeze()[m1]) == 0: val_acc1[i_th] += 0.0
                else: val_acc1[i_th] += torch.count_nonzero(torch.eq(val_pred_classes[m1],val_labels.squeeze()[m1])).item()/len(val_labels.squeeze()[m1])
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        for i_th, threshold in enumerate(np.arange(1,20)*0.05):
            val_acc0[i_th] /= len(val_loader)
            val_acc1[i_th] /= len(val_loader)

        gammas_whole = gammas
        gammas_whole = gammas_whole[gammas_whole>=args.gamma1]
        gammas_whole = gammas_whole[gammas_whole<=args.gamma2]
        print(len(gammas_whole), len(test_loaders))


        np.savetxt(f, np.array([running_loss,val_loss,test_loss,running_acc,val_acc,test_acc,val_acc0[9],test_acc0,val_acc1[9],test_acc1]), newline=" ")
        f.write("\n"); f.close()

        print(f'L={args.L}, set={args.setsize}, {args.gamma1}vs{args.gamma2}, frac={args.fraction}')
        print(f'Tr: {running_loss:.4f}, {running_acc*100:.4f}%')
        print(f'Te: {val_loss:.4f}, {val_acc*100:.4f}%')
        print(f'Te0/1: {val_acc0[9]*100:.4f}%, {val_acc1[9]*100:.4f}%')
    


        if args.test==True:
            phase_diagram_out = np.zeros_like(gammas)
            print(phase_diagram_out.shape)
            phase_diagram_out_err = np.zeros_like(gammas)

            for j,test_loader in enumerate(test_loaders):
                mat_output = []
                for _,testdata in enumerate(test_loader):
                    test_inputs = testdata[0].to(device)
                    test_outputs = model.module(test_inputs)
                    mat_output.extend(test_outputs.detach().cpu().numpy())
                mat_output = np.array(mat_output)
                phase_diagram_out[j] = mat_output.mean(axis=0)
                phase_diagram_out_err[j] = sem(mat_output)
                del mat_output
            np.savetxt(f_out, phase_diagram_out, newline=" ")
            f_out.write("\n")
            np.savetxt(f_out, phase_diagram_out_err, newline=" ")
            f_out.write("\n"); f_out.close()

        torch.cuda.empty_cache()
        print('Done: ', args.setsize, setsizemodel)
    print(f'Time: {time.time()-start} s, {(time.time()-start)/60} min, {(time.time()-start)/3600} hour \n\n')


if __name__ == "__main__":
    main()

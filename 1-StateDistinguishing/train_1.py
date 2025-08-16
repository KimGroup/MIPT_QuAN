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
    parser.add_argument("-n_mini", help = "# of mini set", type = int, default = 1)
    parser.add_argument("-L", help = "L", type = int, default = None)
    parser.add_argument("-init", help = "0 or +", type = int, default = 0)
    parser.add_argument("-gamma", type = float, default = 0.9)
    parser.add_argument("-fraction", type = str, default = '1')
    parser.add_argument("-setsize", help = "setsize", type = int, default = None)
    parser.add_argument("-noise", help = "noise", type = str, default = '')

    parser.add_argument("-prefix", help = "prefix", type = str, default = None)
    parser.add_argument("-saveprefix", help = "save folder prefix", type = str, default = 'g2_saved_models_MIPT')
    parser.add_argument("-epoch", help = "total epochs", type = int, default = None)
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
    else: basic_size = 1000000 // args.setsize * args.setsize
    if n_bitstring+basic_size >= 7498360: sys.exit("ERROR: larger than total sample size")

    model_name = f'{args.L}_g{args.gamma}_0vs{args.fraction}'\
                +f'-{args.modelnum}_set{args.setsize}__h{args.hdim}nh{args.nhead}_ch{args.ch}ker2st1' 
    os.makedirs(f'{args.saveprefix}',exist_ok=True)
    os.makedirs(f'{args.saveprefix}/{args.wandb_name}',exist_ok=True)
        
    f = open(f'{args.saveprefix}/{args.wandb_name}/log_{model_name}.txt', 'w')
    f.close()
    f_val = open(f'{args.saveprefix}/{args.wandb_name}/acc_{model_name}.txt', 'w')
    f_val.close()
    if args.test==True:
        f_out = open(f'{args.saveprefix}/{args.wandb_name}/phase_out_{model_name}.txt', 'w')
        f_out.close()

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


    x1, x2, x3v, x4v = gen_data_wkmsmt_State_embedding(args, args.gamma, args.L, n_bitstring=n_bitstring, basic_size=basic_size, maxnfile = 200)
    x1, x2, x3v, x4v = x1.astype(datatype), x2.astype(datatype), x3v.astype(datatype), x4v.astype(datatype)
    x, y = gen_data_sample(args.setsize, x1, x2, Nt, datatype=datatype)
    train_loader = DataLoader(dataset = training_set(x, y), batch_size=args.batchsize, shuffle=True)
    del x, y
    xt, yt = gen_data_sample(args.setsize, x3v, x4v, Nt, datatype=datatype)
    val_loader = DataLoader(dataset = testing_set(xt, yt), batch_size=args.batchsize * 10, shuffle=True)
    del xt, yt

    if args.wandb_name and not args.debug:
        wandb.init(project='MIPT', name=args.wandb_name)
        wandb.log({'# of parameters': total_params})

    if args.prev != None:
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
    print('model loaded.')

    total_epoch = args.epoch
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=1e-5)
    criterion = nn.BCELoss() #F.binary_cross_entropy

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device); model.to(device)

    best_loss_t = args.batchsize
    
    start = time.time() 
    for epoch in range (total_epoch):
        running_loss, val_loss, test_loss = 0.0, 0.0, 0.0
        val_loss0, val_loss1 = 0.0, 0.0
        running_acc, val_acc, test_acc= 0.0, 0.0, 0.0
        val_acc0, val_acc1 = np.zeros(19), np.zeros(19)
        test_acc0, test_acc1 = 0.0, 0.0
        if (total_epoch-epoch)%args.shuffle_epoch==0:

            x, y = gen_data_sample(args.setsize, x1, x2, Nt, datatype=datatype, verb=False)
            train_loader = DataLoader(dataset = training_set(x, y), batch_size=args.batchsize, shuffle=True)
            del x, y
            xt, yt = gen_data_sample(args.setsize, x3v, x4v, Nt, datatype=datatype, verb=False)
            val_loader = DataLoader(dataset = testing_set(xt, yt), batch_size=args.batchsize * 10, shuffle=True)
            del xt, yt

        if args.logprev == None: 
            f = open(f'{args.saveprefix}/{args.wandb_name}/log_{model_name}.txt', 'a')
            f_val = open(f'{args.saveprefix}/{args.wandb_name}/acc_{model_name}.txt', 'a')
            f_out = open(f'{args.saveprefix}/{args.wandb_name}/phase_out_{model_name}.txt', 'a')
        else: 
            f = open(args.logprev, 'a')
            if args.test==True:
                f_out = open(args.logprev.replace('log_','phase_out_'), 'a')
    
        model.train()
        for i,data in (enumerate(train_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model.module(inputs)
            if epoch==0 and i==0: print(inputs.shape, labels.shape, outputs.shape)
            del inputs
            
            optimizer.zero_grad()
            loss = criterion(outputs.squeeze(), labels.float().squeeze())
            if torch.isnan(loss).any(): break
            loss.backward(); optimizer.step()
            loss.detach().cpu().numpy()
            
            running_loss += loss.item()
            pred_classes = torch.cat([1-outputs, outputs], axis=1).argmax(axis=1)
            running_acc += torch.count_nonzero(torch.eq(pred_classes, labels.squeeze())).item()/len(labels)

        running_loss /= len(train_loader)
        running_acc /= len(train_loader)
        torch.cuda.empty_cache()
        
        # validation
        if total_epoch > 500: epochcriteria = epoch % (total_epoch//500)
        else: epochcriteria = epoch % 1

        if epochcriteria ==0:
            model.eval()
            mat_output0, mat_output1 = [], []
            for _,valdata in enumerate(val_loader):
                val_inputs, val_labels = valdata[0].to(device), valdata[1].to(device)
                val_outputs = model.module(val_inputs)
                
                del val_inputs
                loss_ = criterion(val_outputs.squeeze(), val_labels.float().squeeze())

                m0 = val_labels.squeeze() < 0.5
                m1 = val_labels.squeeze() >= 0.5
                loss_0 = criterion(val_outputs[m0].squeeze(), val_labels[m0].float().squeeze())
                loss_1 = criterion(val_outputs[m1].squeeze(), val_labels[m1].float().squeeze())
                loss_.detach().cpu().numpy()
                loss_0.detach().cpu().numpy()
                loss_1.detach().cpu().numpy()
                mat_output0.extend(val_outputs[m0].detach().cpu().numpy())
                mat_output1.extend(val_outputs[m1].detach().cpu().numpy())
                
                val_pred_classes = torch.cat([1-val_outputs, val_outputs], axis=1).argmax(axis=1)
                val_loss += loss_.item()
                val_loss0 += loss_0.item()
                val_loss1 += loss_1.item()
                val_acc += torch.count_nonzero(torch.eq(val_pred_classes, val_labels.squeeze())).item()/len(val_labels)
                del loss_
                for i_th, threshold in enumerate(np.arange(1,20)*0.05):
                    m0 = val_outputs.squeeze() < threshold
                    m1 = val_outputs.squeeze() >= threshold
                    if len(val_labels.squeeze()[m0]) == 0: val_acc0[i_th] += 0.0
                    else: val_acc0[i_th] += torch.count_nonzero(torch.eq(val_pred_classes[m0],val_labels.squeeze()[m0])).item()/len(val_labels.squeeze()[m0])
                    if len(val_labels.squeeze()[m1]) == 0: val_acc1[i_th] += 0.0
                    else: val_acc1[i_th] += torch.count_nonzero(torch.eq(val_pred_classes[m1],val_labels.squeeze()[m1])).item()/len(val_labels.squeeze()[m1])
            val_loss /= len(val_loader)
            val_loss0 /= len(val_loader)
            val_loss1 /= len(val_loader)
            val_acc /= len(val_loader)
            for i_th, threshold in enumerate(np.arange(1,20)*0.05):
                val_acc0[i_th] /= len(val_loader)
                val_acc1[i_th] /= len(val_loader)
            learningratearg = optimizer.param_groups[0]["lr"]

            if args.wandb_name:
                wandb.log({'epoch': epoch, 'train_loss': running_loss, 'train_acc': running_acc, \
                    'val_loss': val_loss, 'val_acc': val_acc, "val_acc0": val_acc0, "val_acc1": val_acc1})
            if best_loss_t > val_loss:
                torch.save(model.module.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_vl_{model_name}.pth')
                best_loss_t = val_loss
            if epoch>int(total_epoch*0.25): torch.save(model.module.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_e{epoch}_{model_name}.pth')
            
            np.savetxt(f, np.array([running_loss,val_loss,test_loss,running_acc,val_acc,test_acc,val_acc0[9],test_acc0,val_acc1[9],test_acc1]), newline=" ")
            if 'test' in args.modelnum: np.savetxt(f_val, np.array([val_acc0, val_acc1, val_loss0, val_loss1]), newline=" ")
            else: np.savetxt(f_val, np.concatenate([val_acc0, val_acc1]), newline=" ")
            f.write("\n"); f.close()
            f_val.write("\n"); f_val.close()

            print(f'L={args.L}, set={args.setsize}, {args.gamma}, frac={args.fraction} - [#{epoch}/{total_epoch}], lr = {learningratearg}')
            print(f'Tr: {running_loss:.4f}, {running_acc*100:.4f}%')
            print(f'Te: {val_loss:.4f}, {val_acc*100:.4f}%')
            print(f'Te0/1: {val_acc0[9]*100:.4f}%, {val_acc1[9]*100:.4f}%')

            torch.cuda.empty_cache()
    print(f'Time: {time.time()-start} s, {(time.time()-start)/60} min, {(time.time()-start)/3600} hour \n\n')


if __name__ == "__main__":
    main()

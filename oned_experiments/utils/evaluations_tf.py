import torch
import torchvision
import torch.backends.cudnn as cudnn
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import argparse 
import os

sys.path.append('./')
sys.path.append('./datasets')
sys.path.append('./models')

import seqae
import models.base_networks as bn 

import models
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from einops import rearrange
from sklearn.metrics import r2_score
import pdb
from einops import rearrange
from utils import evaluations as ev
from utils import notebook_utils as nu
from utils import character_analysis as ca
import pickle

from torch.utils.tensorboard import SummaryWriter

import copy

import csv
import ast
from utils import yaml_utils as yu


'''
Scripts for running evaluations and outputting the results to tensorboard readable format.
'''



def write_tf(targdir, root='../result/nfa'):
    device = 0
    targdir_path = os.path.join(root, targdir)

    writer = SummaryWriter(log_dir=targdir_path)

    tp=18

    Mlist = []
    config = nu.load_config(targdir_path)
    logs = nu.read_log(targdir_path)

    dataconfig = config['train_data']
    dataconfig['args']['T'] = config['T_cond'] + tp
    dataconfig['args']['test'] = 1


    data = yu.load_component(dataconfig)
    train_loader = DataLoader(
            data, batch_size=config['batchsize'], shuffle=False, num_workers=config['num_workers'])

    torch.manual_seed(0)
    train_loader = DataLoader(data, batch_size=config['batchsize'], shuffle=True, num_workers=config['num_workers'])
    model_config = config['model']

    model_config['args']['dim_data'] = config['train_data']['args']['N']

    model = yu.load_component(model_config)
    iterlist = nu.iter_list(targdir_path)
    maxiter = np.max(nu.iter_list(targdir_path))
    nu.load_model(model, targdir_path, maxiter)

    print("Creating the example images")
    #images
    writer_images(train_loader, model, config, device, tp, writer)

    print("Prediction evaluation in process...") 
    #error evaluation
    allresults, targ, xnext  = ev.prediction_evaluation([targdir_path], device =0, n_cond=2, 
                                                    tp=tp, repeats=3,predictive=False,
                                                    reconstructive = False,alteration={},
                                                   mode='notebook')

    print("Conducting character analysis...")
    #character_analysis
    rholist = allresults['Ms'][targdir_path].detach()
    gs = torch.flatten(allresults['labels'][targdir_path])
    targfreq, prods = ca.inner_prod(rholist, gs, maxfreq=64, bins=65)
    writer_spectrum(targfreq, prods, config, data, writer)

    print("Reporting the prediction error...")
    #prediction error
    pred_error = allresults['results'][targdir_path][0]
    for i in range(len(pred_error)):
        writer.add_scalar("prediction", pred_error[i], i)
    writer.close()    


def writer_spectrum(targfreq, prods, config, data, writer):
    plt.figure()
    plt.plot(targfreq, prods, label='learnt')
    deltas = ca.deltafxn(targfreq, data.freqsel)*2
    plt.plot(targfreq, deltas, label='gt')

    plt.title(f"""Frequencies learnt from datasets with Freq:{data.freqsel}""")
    plt.legend()

    imgpath = os.path.join(config['log_dir'], 'pltspec.png')
    prodpath = os.path.join(config['log_dir'], 'prods.pkl')

    prodResults = {} 
    prodResults['truefreq'] = data.freqsel
    prodResults['targfreq'] = targfreq
    prodResults['prods'] = prods
    prodResults['deltas'] = deltas

    with open(prodpath, 'wb') as handle:
        pickle.dump(prodResults, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.savefig(imgpath)
    myimage = torchvision.io.read_image(imgpath)
    myimage = myimage[:3, :, :]
    writer.add_image('spectrum', myimage, 0)

 

def writer_images(train_loader, model, config, device, tp, writer):
 
    images, label = next(iter(train_loader))
    model.eval().to(device)
    if type(images) == list:
        images = torch.stack(images)
        images = images.transpose(1, 0)
    images = images.to(device).float()

    reconst = False
    regconfig = config['reg']
    #loss,  loss_dict = model.loss(images,  T_cond=config['T_cond'], return_reg_loss=True, reconst=reconst, regconfig=regconfig)
    T_cond = config['T_cond']
    xs = images
    return_reg_loss = False
    xs_cond = xs[:, :T_cond]
    xs_pred = model(xs_cond, return_reg_loss=return_reg_loss,
                        n_rolls=xs.shape[1] - T_cond, reconst=reconst, indices=label).to('cpu')
    xs_target = xs[:, T_cond:].to('cpu')

    check_indices = [0,1,2]
    graphtime = 5
    plt.figure(figsize=(20,2*len(check_indices)))
    current_idx = 1

    for check_idx in check_indices:
        for k in range(graphtime):
            plt.subplot(len(check_indices), graphtime, current_idx)
            plt.plot(xs_target[check_idx][k])
            plt.plot(xs_pred[check_idx][k].detach())
            current_idx = current_idx + 1
        
    imgpath = os.path.join(config['log_dir'], 'pltimg.png')
    plt.savefig(imgpath)
    myimage = torchvision.io.read_image(imgpath)
    myimage = myimage[:3, :, :]
    writer.add_image('visuals', myimage, 0)
    


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--targdir', type=str)
    parser.add_argument('--all', type=int, default=0)
    parser.add_argument('--root', type=str, default='../result/nfa')


    args = parser.parse_args()

    if args.all == 0:
        write_tf(args.targdir)
    else:
        dirlist = os.listdir(os.path.join(args.root, args.targdir))
        for subdir in dirlist:
            targpath = os.path.join(args.targdir,subdir)
            write_tf(targpath, root=args.root)



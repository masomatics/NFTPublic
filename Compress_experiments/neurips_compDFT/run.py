import os
import argparse
import yaml
import copy
import functools
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
from utils import yaml_utils as yu


def train(config):

    torch.cuda.empty_cache()
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    if torch.cuda.is_available():
        device = torch.device('cuda',config['gpu_id'])
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        gpu_index = -1

    print(device)
    
    # Dataaset
    data = yu.load_component(config['train_data'])
    
    train_loader = DataLoader(
        data, batch_size=config['batchsize'], shuffle=not config['fixedM'], num_workers=config['num_workers'])

    # Def. of Model and optimizer
    model = yu.load_component(config['model'])
    optimizer = torch.optim.Adam(model.parameters(), config['lr']) 

    try:
        if config['model']['args']['transition_model'] not in ('LS', 'Fixed',  'AbelMSP'):
            raise ValueError("transition model must be in ('LS', 'Fixed', 'AbelMSP')")
    except ValueError as ve:
        print("Error:", ve)
        exit(1)



    manager = ppe.training.ExtensionsManager(
        model, optimizer, None,
        iters_per_epoch=len(train_loader),
        out_dir=config['log_dir'],
        stop_trigger=(config['max_iteration'], 'iteration')
    )

    manager.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'train/loss', 'train/reconst', 'train/pred', 'train/testloss', 'train/loss_bd']),
        trigger=(config['report_freq'], 'iteration'))
    manager.extend(extensions.LogReport(
        trigger=(config['report_freq'], 'iteration')))
    manager.extend(
        extensions.snapshot(
            target=model, filename='snapshot_model_iter_{.iteration}'),
        trigger=(config['model_snapshot_freq'], 'iteration'))
    manager.extend(
        extensions.snapshot(
            target=manager, filename='snapshot_manager_iter_{.iteration}', n_retains=1),
        trigger=(config['manager_snapshot_freq'], 'iteration'))
    
    # figure path core
    trans=config['model']['args']['transition_model']
    if config['model']['args']['second_transition'] is not None:
            trans=config['model']['args']['second_transition']
    da=config['model']['args']['dim_a']
    dm=config['model']['args']['dim_m']
    nfq=config['train_data']['args']['nfreq']
    sd=config['seed']
    maxitr=config['max_iteration']
    fpathcore = 'DFT_{}_da{}_dm{}_nfq{}_sd{}_{}'.format(trans,da,dm,nfq,sd,maxitr)
    
    # Run training loop
    print("Start training...")
    yu.load_component_fxn(config['training_loop'])(
        manager, model, optimizer, train_loader, config, device, fpathcore=fpathcore)
    



if __name__ == '__main__':
    # Loading the configuration arguments from specified config path
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--fig_dir', type=str)
    parser.add_argument('-a', '--attrs', nargs='*', default=())
    parser.add_argument('-w', '--warning', action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['config_path'] = args.config_path
    config['fig_dir'] = args.fig_dir
    config['log_dir'] = args.log_dir

    print(config['config_path']) 


    # Modify the yaml file using attr
    for attr in args.attrs:
        module, new_value = attr.split('=')
        keys = module.split('.')
        target = functools.reduce(dict.__getitem__, keys[:-1], config)
        if keys[-1] in target.keys():
            target[keys[-1]] = yaml.safe_load(new_value)
        elif keys[-1]=='nfreq':
            config['train_data']['args']['nfreq'] = int(new_value)
        elif keys[-1]=='shift_range':
            config['train_data']['args']['shift_range'] = float(new_value)
        elif keys[-1]=='dec_hdim':
            config['model']['args']['dec_hdim'] = int(new_value)
        elif keys[-1]=='coef_internal':
            config['model']['args']['coef_internal'] = float(new_value)
        else:
            raise ValueError('The following key is not defined in the config file:{}', keys)
        
    for k, v in sorted(config.items()):
        print("\t{} {}".format(k, v))

    # create the result directory and save yaml
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    _config = copy.deepcopy(config)
    configpath = os.path.join(config['log_dir'], "config.yml")
    open(configpath, 'w').write(
        yaml.dump(_config, default_flow_style=False)
    )


    # Training
    train(config)

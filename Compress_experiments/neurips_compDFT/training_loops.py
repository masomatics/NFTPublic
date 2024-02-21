import math
import torch
from torch import nn
import pytorch_pfn_extras as ppe
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from period_fn import Shifted_FreqFun



# training loop with switch of the transition model
def loop_seqmodel_sw(manager, model, optimizer, train_loader, config, device, fpathcore=None):
    # setting lr in gradient descent
    lr = config['lr']
    lr_decay_ratio1 = 3.0   # defulat value
    lr_decay_ratio2 = 1.0   # defulat value
    if config['training_loop']['args']['lr_decay_ratio'] is not None:
        lr_decay_ratio1 = config['training_loop']['args']['lr_decay_ratio']
        lr_decay_ratio2 = 1.0
        lr_decay_iter1 = config['training_loop']['args']['lr_decay_iter']
        lr_decay_iter2 = config['training_loop']['args']['lr_decay_iter']
    if config['training_loop']['args']['lr_decay_iter1'] is not None:
        lr_decay_ratio1 = config['training_loop']['args']['lr_decay_ratio1']
        lr_decay_ratio2 = 1.0
        lr_decay_iter1 = config['training_loop']['args']['lr_decay_iter1']
        lr_decay_iter2 = config['training_loop']['args']['lr_decay_iter1']
    if config['training_loop']['args']['lr_decay_iter2'] is not None:
        lr_decay_ratio2 = config['training_loop']['args']['lr_decay_ratio2']
        lr_decay_iter2 = config['training_loop']['args']['lr_decay_iter2']

    testflag = False  


    while not manager.stop_trigger:
        for datapair in train_loader:
            with manager.run_iteration():
                reconst = True if manager.iteration < config['training_loop']['args']['reconst_iter'] else False
                
                # setting learning rate
                if manager.iteration >= lr_decay_iter1:
                    eta = lr/lr_decay_ratio1
                if manager.iteration >= lr_decay_iter2:
                    eta = lr/(lr_decay_ratio2*lr_decay_ratio1)
                else:
                    eta = config['lr']
                optimizer.param_groups[0]['lr'] = eta   # setting learning rate
                
                # switch transition model
                if manager.iteration == config['training_loop']['args']['switch_iter']:
                    model.switch_transition_model(model.second_transition)
                    
                    for param in model.enc.phi.parameters():
                        param.requires_grad = False
                    for param in model.dec.lin2.parameters():
                        param.requires_grad = False
                    for param in model.dec.lin3.parameters():
                        param.requires_grad = False

                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=eta)


                model.train()
                
                # set indata for the input to learning (images, g_true, dataidx)
                indata = []
                images = torch.stack(datapair[0]).transpose(1, 0).to(device,torch.float32)
                indata.append(images)

                if config['train_data']['args']['shift_label']:
                    g_true = datapair[1].to(device,torch.float32)
                    indata.append(g_true)
                    model.g_true = g_true
                else:
                    g_true=None

                if config['train_data']['args']['indexing']:
                    dataidx = datapair[2].to(device,torch.float32)      # one-hot vector to specify the data
                    indata.append(dataidx)
                else:
                    dataidx=None

                loss, loss_reg = model.loss(
                        indata[0], 
                        gelement=g_true, 
                        T_cond=config['T_cond'], 
                        return_reg_loss=True, 
                        reconst=reconst,
                        return_g=model.return_g
                )
                # loss is used for backprop, while loss_reg is to show the results.
                (loss_bd, loss_orth, loss_internal_T, loss_reconst, loss_pred)=loss_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #trainable_parameter_names = []
                #for name, param in model.named_parameters():
                #    if param.requires_grad:
                #        trainable_parameter_names.append(name)



                if manager.iteration ==config['max_iteration']-1 and testflag:
#                if testflag:
                    
                    test_loss, pred_x, reg_loss, g_est = model.testing(
                        indata[0],
                        T_cond=config['T_cond'],
                        return_reg_loss=True,
                        n_rolls=images.shape[1]-config['T_cond'],
                        predictive=True,
                        reconst=True,
                        return_g=model.return_g,
                        gelement=g_true
                    )
                    g_err = torch.mean((g_est.squeeze()-model.g_true)**2)
                    
                    dima = model.dim_a
                    dimm = model.dim_m
                    he = model.dim_he
                    hd = model.dim_hd
                    nfreq = config['train_data']['args']['nfreq']
                    directory_path =  config['fig_dir']
                    x_preds_rollout = model.rollout(images,T_cond=config['T_cond'], n_rolls=images.shape[1]-config['T_cond'],predictive=True,reconst=True,gelement=g_true)

                    fpath_rollout='{}rollout_train_{}.png'.format(directory_path,fpathcore)
                    dt = 1.0/images.shape[-1]
                    t = np.arange(0, images.shape[-1]*dt, dt) # time domain 
                    model.plot_rollout(
                        fpath_rollout,
                        x_preds_rollout[0,:],
                        images[0,:],
                        t,
                        col_true='b-',
                        col_est='r-',
                        )
                
                    test_data = Shifted_FreqFun( 
                        Ndata=16,   # Number of data
                        N=128,          # Number of observation points 
                        T=5,            # Time steps in a sequence
                        shift_label=True,
                        indexing=False,
                        fixedM=False,
                        batchM_size=16,
                        shift_range=2,  # frequency N/(5*shift_range)
                        max_shift=[0.0, 2 * math.pi / 2],   # range of shift action
                        max_T=5,
                        shared_transition=False,
                        rng=None,
                        nfreq=12,        # Number of selected frequecy to make a function
                        coef=None,      # Coefficients of the selected frequencies. 
                        ns=0.0,          # Noise level of additive Gaussian noise
                        shifts = None,
                        random_shifts = False   # If true, each data has different shifts over epochs
                    )
                    test_loader = DataLoader(
                        test_data, batch_size=16, shuffle=not config['fixedM'], num_workers=config['num_workers'])
                    
                    for test_datapair in test_loader:
                        model.eval()
                
                        # set indata for the input to learning (images, g_true, dataidx)
                        test_indata = []
                        test_images = torch.stack(test_datapair[0]).transpose(1, 0).to(device,torch.float32)
                        test_indata.append(test_images)

                        
                        g_true = test_datapair[1].to(device,torch.float32)
                        test_indata.append(g_true)
                        model.g_true = g_true


                    testtest_loss, test_pred_x, test_g_loss, test_g_est = model.testing(
                        test_images,
                        T_cond=config['T_cond'],
                        return_reg_loss=True,
                        n_rolls=test_images.shape[1]-config['T_cond'],
                        predictive=True,
                        reconst=True,
                        return_g=model.return_g,
                        gelement=g_true
                    )
                    test_g_err = torch.mean((test_g_est.squeeze()-model.g_true)**2)

                    testx_preds_rollout = model.rollout(test_images,T_cond=config['T_cond'], n_rolls=test_images.shape[1]-config['T_cond'],predictive=True,reconst=True,gelement=g_true)

                    fpath_rollout='{}rollout_test_{}.png'.format(directory_path,fpathcore)
                    dt = 1.0/images.shape[-1]
                    t = np.arange(0, images.shape[-1]*dt, dt) # time domain 
                    model.plot_rollout(
                        fpath_rollout,
                        testx_preds_rollout[0,:],
                        test_images[0,:],
                        t,
                        col_true='b-',
                        col_est='r-',
                        )
                                                     
                    ppe.reporting.report({
                    'train/testloss': test_loss.item(),
                    'train/test_internal': reg_loss.item(),
                    })                  
            
                    ppe.reporting.report({
                    'train/g_err': g_err.item(),
                    })

                ppe.reporting.report({
                    'train/loss': loss.item(),
                    'train/reconst': loss_reconst.item(),
                    'train/pred': loss_pred.item(),
                    'train/loss_bd': loss_bd.item(),
                    'train/loss_orth': loss_orth.item(),
                })          
            
            if manager.stop_trigger:
                make_figs(model,images,config,fpathcore)
               
                break




def loop_seqmodel(manager, model, optimizer, train_loader, config, device, fpathcore=None):
    # setting lr in gradient descent
    lr = config['lr']
    lr_decay_ratio1 = 3.0   # defulat value
    lr_decay_ratio2 = 1.0   # defulat value
    if config['training_loop']['args']['lr_decay_ratio'] is not None:
        lr_decay_ratio1 = config['training_loop']['args']['lr_decay_ratio']
        lr_decay_ratio2 = 1.0
        lr_decay_iter1 = config['training_loop']['args']['lr_decay_iter']
        lr_decay_iter2 = config['training_loop']['args']['lr_decay_iter']
    if config['training_loop']['args']['lr_decay_iter1'] is not None:
        lr_decay_ratio1 = config['training_loop']['args']['lr_decay_ratio1']
        lr_decay_ratio2 = 1.0
        lr_decay_iter1 = config['training_loop']['args']['lr_decay_iter1']
        lr_decay_iter2 = config['training_loop']['args']['lr_decay_iter1']
    if config['training_loop']['args']['lr_decay_iter2'] is not None:
        lr_decay_ratio2 = config['training_loop']['args']['lr_decay_ratio2']
        lr_decay_iter2 = config['training_loop']['args']['lr_decay_iter2']

    testflag = False  

    while not manager.stop_trigger:
        for datapair in train_loader:
            with manager.run_iteration():
                reconst = True if manager.iteration < config['training_loop']['args']['reconst_iter'] else False
                if manager.iteration >= lr_decay_iter1:
                    eta = lr/lr_decay_ratio1
                if manager.iteration >= lr_decay_iter2:
                    eta = lr/(lr_decay_ratio2*lr_decay_ratio1)
                else:
                    eta = config['lr']
                optimizer.param_groups[0]['lr'] = eta   # setting learning rate
                if model.transition_model == 'MgDGD':   # setting lr in gradient descent
                    model.set_eta(0.0003)      # for MAML
                

                
                model.train()
                
                # set indata for the input to learning (images, g_true, dataidx)
                indata = []
                images = torch.stack(datapair[0]).transpose(1, 0).to(device,torch.float32)
                indata.append(images)

                if config['train_data']['args']['shift_label']:
                    g_true = datapair[1].to(device,torch.float32)
                    indata.append(g_true)
                    model.g_true = g_true
                else:
                    g_true=None

                if config['train_data']['args']['indexing']:
                    dataidx = datapair[2].to(device,torch.float32)      # one-hot vector to specify the data
                    indata.append(dataidx)
                else:
                    dataidx=None

                loss, loss_reg = model.loss(
                        indata[0], 
                        gelement=g_true, 
                        T_cond=config['T_cond'], 
                        return_reg_loss=True, 
                        reconst=reconst,
                        return_g=model.return_g,
                    )   
                (loss_bd, loss_orth, loss_internal_T, loss_reconst, loss_pred)=loss_reg

                # loss is used for backprop, while loss_reg is to show the results.
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if manager.iteration ==config['max_iteration']-1 and testflag:
#                if testflag:
                    
                    test_loss, pred_x, reg_loss, g_est = model.testing(
                        indata[0],
                        T_cond=config['T_cond'],
                        return_reg_loss=True,
                        n_rolls=images.shape[1]-config['T_cond'],
                        predictive=True,
                        reconst=True,
                        return_g=True,
                        gelement=g_true
                    )
                    g_err = torch.mean((g_est.squeeze()-model.g_true)**2)
                    
                    dima = model.dim_a
                    dimm = model.dim_m
#                    he = model.dim_he
#                    hd = model.dim_hd
                    nfreq = config['train_data']['args']['nfreq']
                    directory_path =  config['fig_dir']
                    x_preds_rollout = model.rollout(images,T_cond=config['T_cond'], n_rolls=images.shape[1]-config['T_cond'],predictive=True,reconst=True,gelement=g_true)

                    fpath_rollout='{}rollout_train_{}.png'.format(directory_path,fpathcore)
                    dt = 1.0/images.shape[-1]
                    t = np.arange(0, images.shape[-1]*dt, dt) # time domain 
                    model.plot_rollout(
                        fpath_rollout,
                        x_preds_rollout[0,:],
                        images[0,:],
                        t,
                        col_true='b-',
                        col_est='r-',
                        )                  
                    
                    test_data = Shifted_FreqFun( 
                        Ndata=16,   # Number of data
                        N=128,          # Number of observation points 
                        T=5,            # Time steps in a sequence
                        shift_label=True,
                        indexing=False,
                        fixedM=False,
                        batchM_size=16,
                        shift_range=2,  # frequency N/(5*shift_range)
                        max_shift=[0.0, 2 * math.pi / 2],   # range of shift action
                        max_T=5,
                        shared_transition=False,
                        rng=None,
                        nfreq=12,        # Number of selected frequecy to make a function
                        coef=None,      # Coefficients of the selected frequencies. 
                        ns=0.0,          # Noise level of additive Gaussian noise
                        shifts = None,
                        random_shifts = False   # If true, each data has different shifts over epochs
                    )
                    test_loader = DataLoader(
                        test_data, batch_size=16, shuffle=not config['fixedM'], num_workers=config['num_workers'])
                    
                    for test_datapair in test_loader:
                        model.eval()
                
                        # set indata for the input to learning (images, g_true, dataidx)
                        test_indata = []
                        test_images = torch.stack(test_datapair[0]).transpose(1, 0).to(device,torch.float32)
                        test_indata.append(test_images)

                        
                        g_true = test_datapair[1].to(device,torch.float32)
                        test_indata.append(g_true)
                        model.g_true = g_true


                    testtest_loss, test_pred_x, test_g_loss, test_g_est = model.testing(
                        test_images,
                        T_cond=config['T_cond'],
                        return_reg_loss=True,
                        n_rolls=test_images.shape[1]-config['T_cond'],
                        predictive=True,
                        reconst=True,
                        return_g=model.return_g,
                        gelement=g_true
                    )
                    test_g_err = torch.mean((test_g_est.squeeze()-model.g_true)**2)

                    testx_preds_rollout = model.rollout(test_images,T_cond=config['T_cond'], n_rolls=test_images.shape[1]-config['T_cond'],predictive=True,reconst=True,gelement=g_true)

                    fpath_rollout='{}rollout_test_{}.png'.format(directory_path,fpathcore)
                    dt = 1.0/images.shape[-1]
                    t = np.arange(0, images.shape[-1]*dt, dt) # time domain 
                    model.plot_rollout(
                        fpath_rollout,
                        testx_preds_rollout[0,:],
                        test_images[0,:],
                        t,
                        col_true='b-',
                        col_est='r-',
                        )
                                                      
                    ppe.reporting.report({
                    'train/testloss': test_loss.item(),
                    'train/test_internal': reg_loss.item(),
                    })
                                
                    ppe.reporting.report({
                    'train/g_err': g_err.item(),
                    })

                ppe.reporting.report({
                    'train/loss': loss.item(),
                    'train/reconst': loss_reconst.item(),
                    'train/pred': loss_pred.item(),
                    'train/loss_bd': loss_bd.item(),
                })          
            
            if manager.stop_trigger:
                make_figs(model,images,config,fpathcore)              
                break



# for fft graphics    
def make_figs(model,images,config,fpathcore):
    directory_path=config['fig_dir']
    t_cond = config['T_cond']
    N = images.shape[2]
    dt = 1.0/N
    t = np.arange(0, N*dt, dt) # time domain 
    f_true = images.cpu()   #    f_true[:,0,:]: reconst, f_true[:,2,:]: prediction
    f_true = f_true.clone().detach().numpy()
    xs_cond = images[:, :t_cond]
    
    # reconstruction and prediction
    x_preds, _, losses = model(xs_cond, gelement=model.g_true, return_reg_loss=True, n_rolls=1, predictive=True, reconst=True)
    f_reconst = x_preds[:,0,:].cpu()
    f_reconst = f_reconst.clone().detach().numpy()
    f_pred = x_preds[:,2,:].cpu()
    f_pred = f_pred.clone().detach().numpy()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    cs=colors
    plt.style.use('ggplot')

    fig_reconst = plt.figure(figsize=[18, 3])
    for i in range(4):
        ax = fig_reconst.add_subplot(1, 4, i+1)
        ax.plot(t, f_true[i,0,:], 'r-', t, f_reconst[i,:],'b-')
        plt.xlabel('time', fontsize=16)
        plt.ylabel('values', fontsize=16)

    # Check if the directory does not already exist
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Figures written in '{directory_path}'.")

    plt.title('Reconstruction') 
    plt.plot()
    plt.tight_layout()        
    plt.savefig('{}Reconst_{}.png'.format(directory_path,fpathcore))


    fig_pred = plt.figure(figsize=[18, 3])
    for i in range(4):
        ax = fig_pred.add_subplot(1, 4, i+1)
        ax.plot(t, f_true[i,2,:], 'r-', t, f_pred[i,:],'b-')
        plt.xlabel('time', fontsize=16)
        plt.ylabel('values', fontsize=16)

    plt.title('Predcition') 
    plt.plot()
    plt.tight_layout()
    plt.savefig('{}Pred_{}.png'.format(directory_path,fpathcore))

    plt.close('all')



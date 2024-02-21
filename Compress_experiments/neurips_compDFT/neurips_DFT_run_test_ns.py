import os
import sys
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
from utils import yaml_utils as yu
from period_fn import Shifted_FreqFun, Shifted_FreqFun_nl, Shifted_FixedFreqFun_nl


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config 


def load_model(model, log_dir, iters):
    model.load_state_dict(torch.load(os.path.join(
        log_dir, 'snapshot_model_iter_{}'.format(iters)), map_location=device))

def angle_dist(g1, g2):
    # L2 distance mod 2pi 
    diff = g1 - g2
    # Apply modulo 2π to the difference
    diff_mod_2pi = torch.fmod(diff + 2 * torch.pi, 2 * torch.pi)  
    # Choose the smallest difference between the original and the mod 2π difference
    distance = torch.min(diff_mod_2pi, 2 * torch.pi - diff_mod_2pi)
    return distance**2


def band_limited_dft(X, n_frequencies):
    # Perform the FFT
    fft_X = np.fft.fft(X)

    # Apply the low-pass filter by keeping only the specified number of frequencies
    filtered_fft_X = np.zeros_like(fft_X)
    filtered_fft_X[:n_frequencies] = fft_X[:n_frequencies]

    # Perform the inverse FFT to reconstruct the signal
    reconstructed_X = np.fft.ifft(filtered_fft_X)

    # Since the original signal is real, return only the real part of the reconstructed signal
    return reconstructed_X.real


  
# Plot function graphs  
def plot_graphs(fpath,f_true,f_est,x,title,col_true='r-',col_est='b-'):
    Nfig = 4   
    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #colors = prop_cycle.by_key()['color']
    #cs=colors
    plt.style.use('ggplot')
              
    fig = plt.figure(figsize=[18, 3])
    for i in range(Nfig):
        ax = fig.add_subplot(1, Nfig, i+1)
        ax.plot(x, f_true[i,:], col_true, x, f_est[i,:],col_est)
        #plt.xlabel('time', fontsize=16)
        if i==0:
            plt.ylabel('values', fontsize=20)
        # plt.legend(fontsize=18)  
    plt.suptitle(title, fontsize=24)

 
    plt.plot()
    plt.tight_layout()         
    plt.savefig(fpath)
    plt.clf()
    plt.close()


    
class Result:
    def __init__(self,result_max,result_same,dim_a,nfreq,test_freq,shift_range):
        self.result_max = result_max
        self.result_same = result_same
        self.dim_a = dim_a
        self.nfreq = nfreq
        self.test_freq = test_freq
        self.shift_range = shift_range
        

def setarg(num):
    num_args = len(sys.argv) - 1

    if num_args < num:
        print("the number of arguments ({}) are less than expected '{})".format(num_args,num))
    else:
        transition_model = sys.argv[1]
        dim_a = int(sys.argv[2])
        dim_m = int(sys.argv[3])
        datagen = sys.argv[4]
        ns = float(sys.argv[5])
        logdir_core = sys.argv[6]
        fig_path = sys.argv[7]
    
    return transition_model,dim_a,dim_m,datagen,ns,logdir_core,fig_path




#####################################################################
#  main test routine
#



if __name__ == '__main__':

    
    transition_model,dim_a,dim_m,datagen,ns,logdir_core,fig_path = setarg(7)

    if ns==0.0:
        DFT_compute = True
    else:
        DFT_compute = False

    if transition_model == 'Fixed':
        trans = 'g-NFT'
    elif transition_model == 'AbelMSP':
        trans = 'AbelG-NFT'


    # dictionary for data generation function
    if datagen not in ['linear', 'nonlinear', 'nl_fixed', 'linear_fixed']:
        print('datagen must be either linear or nonlinear')
        sys.exit()

    Datagen_func = {
        'linear': Shifted_FreqFun,
        'nonlinear': Shifted_FreqFun_nl,
        'nl_fixed': Shifted_FixedFreqFun_nl,
        'linear_fixed': Shifted_FreqFun
    }


    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        gpu_index = -1

    nfreq = 5   # temporary setting
    dft_nfreq = 16      # number of frequency components in DFT

    #  Generate test data

    seed_list = range(10)     # seed in training
    Ndata = 5000              # number of test data
    N = 128
    T = 5       # length of each test sequence

    return_g = False
    shift_label = True  # shift label for testing required
    random_shifts = False
    indexing = False
    
    batchM_size = 16

    Nroll = 3
    data_no = 2

    dt = 1.0/N
    t = np.arange(0, N*dt, dt) # time domain 

    random.seed(1234)         # reset random number to generate independent test data
    np.random.seed(1234)  

    # Making test data, which are independent of training data
    #print('Testing ... ')
    bsize = batchM_size       
    test_data = Datagen_func[datagen](
            Ndata=Ndata,   # Number of data
            N=N,          # Number of observation points 
            T=T,            # Time steps in a sequence
            shift_label=shift_label,    # add shift_label (group element g=shift) or not 
            batchM_size=batchM_size,
            indexing=indexing,          # 
            random_shifts=random_shifts,
            nfreq=nfreq,       # Number of selected frequecy to make a function
            shift_range=2,
            ns=0.0,     # evaluation with noiseless data        
        )               # Get training data
    
    test_loader = DataLoader(test_data, bsize, False, num_workers=0)       

    results=[]      
    tbl_losses=np.zeros(len(seed_list))
    tbl_g_losses=np.zeros(len(seed_list))


    test_X = np.zeros((Ndata,N))
    for i in range(Ndata):
        test_X[i,:] = test_data.__getitem__(i)[0][0]


    # Test of NFT for each seed
    for s, seed in enumerate(seed_list):
        logdir="{}sd{}-ns{}".format(logdir_core,seed+1,ns)
        config = load_config(os.path.join(logdir, 'config.yml'))
        #print(config)
        #print('\n')
        #print(logdir)
        model = yu.load_component(config['model'])
        load_model(model, logdir, iters=config['max_iteration'])      # loading trained model
        device = model.device
        model.to(device)
        model.eval()
        reconst = True          # Reconstruction loss is included to prediction loss

        # Set your desired directory path
        dima=config['model']['args']['dim_a']   
        t_cond = config['T_cond']     


        losses=[]
        g_losses=[]
        with torch.no_grad():
            for k, datapair in enumerate(test_loader):
                # set indata for the input to learning (images, g_true, dataidx)
                indata = []
                images = torch.stack(datapair[0]).transpose(1, 0).to(device,torch.float32)
                indata.append(images)

                if shift_label:
                    g_true = datapair[1].to(device,torch.float32)
                    indata.append(g_true)
                else:
                    g_true=None

                if indexing:
                    dataidx = datapair[2].to(device,torch.float32)      # one-hot vector to specify the data
                    indata.append(dataidx)
                else:
                    dataidx=None
                
                if model.transition_model == 'Fixed':
                    gelement = g_true
                else:
                    gelement = None
                
                return_val = model.testing(
                        indata[0],
                        T_cond=config['T_cond'],
                        return_reg_loss=True,
                        n_rolls=images.shape[1]-config['T_cond'],
                        predictive=True,
                        reconst=True,
                        return_g=return_g,
                        gelement = gelement
                    )  
                if return_g:
                    test_loss, pred_x, reg_loss, g_est = return_val
                    g_err = angle_dist(g_est, g_true.unsqueeze(1)).squeeze()
                    g_err = g_err.to('cpu').detach().numpy()
                else:
                    test_loss, pred_x, reg_loss = return_val
                    g_est = None
                    g_err = [0.0]
                            
                
                losses.append(test_loss.to('cpu').detach().numpy())
                g_losses = np.concatenate((g_losses, g_err))  



        g_losses = np.stack(g_losses)
        losses = np.stack(losses)
        tbl_losses[s] = np.mean(losses)
        tbl_g_losses[s] = np.mean(g_losses)
        
        f_est = pred_x[:,0,:].to('cpu').detach().numpy()   #    f_est[:,0,:]: reconst, f_est[:,1,:]: prediction. Note that preds uses images[:,1] as the original in reconstuction and prediction
        # for plotting graphs.  
        # The last minibatch is used for graphics!!
        # True functions 
        nfreq=config['train_data']['args']['nfreq']
        dima=config['model']['args']['dim_a']        
        t_cond = config['T_cond']     
        f_true = images[:,0,:].cpu()   #    f_true[:,0,:]: reconst, f_true[:,1,:]: prediction. Note that preds uses images[:,1] as the original in reconstuction and prediction
        f_true = f_true.clone().detach().numpy()               
        title = 'NFT {} (dim_a = {} dim_m = {})'.format(transition_model,dim_a,dim_m)
        plot_graphs('{}/NFT_test_{}_{}_ns{}.png'.format(fig_path,transition_model,datagen,ns),f_true,f_est,t,title,col_true='r-',col_est='b-')



    
    nft_mean = np.mean(tbl_losses)
    nft_std = np.std(tbl_losses)
    print('\n')    
    print('{}: Data {} ns={} '.format(trans,datagen,ns))    
    print('MSE: {} (STD: {})'.format(nft_mean,nft_std))
    np.set_printoptions(precision=7) 
    np.set_printoptions(linewidth=np.inf)
    print(tbl_losses) 

    if DFT_compute:
        dft_table = np.zeros(Ndata)
        f_true=np.zeros((4,N))
        f_est=np.zeros((4,N))
        print('\nDFT: ')
        for i in range(Ndata):
            X = test_X[i,:]
            Y = 2.0 * band_limited_dft(X,dft_nfreq)
            dft_table[i] = sum((X-Y)**2)
            if i < 4:
                f_true[i,:] = X
                f_est[i,:] = Y

        dft_mse = np.mean(dft_table)

        print('DFT Nfrq = {}  MSE: {}'.format(dft_nfreq,dft_mse))
        title = 'DFT (Num freq = {} in complex)'.format(dft_nfreq)
        plot_graphs('{}/LP_DFT_test_{}_ns{}.png'.format(fig_path,dft_nfreq,datagen,ns),f_true,f_est,t,title,col_true='r-',col_est='b-')



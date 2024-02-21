import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from models.dynamics_models import LinearTensorDynamics
from models.base_networks import MLPEncoder, MLPDecoder

from einops import rearrange, repeat


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def _get_act_module(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'sin':
        return Sin()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


class SeqAETSQmlp(nn.Module):
    def __init__(
            self,
            dim_a,
            dim_m,
            transition_model,
            dim_data=128,
            act='relu',
            depth=2,
            alignment=False,
            change_of_basis=False,
            predictive=True,            
            gpu_id=0,
            internal_loss = False,
            optMg_loss = False,
            enc_hdim = 256,
            dec_hdim = 256,
            coef_internal = 1.0,
            return_g = False,
            second_transition = None
            ):
        super().__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.dim_he = enc_hdim
        self.dim_hd = dec_hdim
        self.predictive = predictive
        if torch.cuda.is_available():
            self.device = torch.device("cuda",gpu_id) 
        else:
            self.device = torch.device("cpu")
        self.enc = MLPEncoder(
            dim_latent=dim_a*dim_m, 
            dim_hidden=enc_hdim, 
            act=_get_act_module(act), 
            depth=depth).to(self.device)
        self.dec = MLPDecoder(
            dim_latent=dim_a*dim_m,
            dim_data=dim_data,
            dim_hidden=dec_hdim, 
            act=_get_act_module(act), 
            depth=depth).to(self.device)
        self.transition_model = transition_model
        self.second_transition = second_transition
        self.dynamics_model = LinearTensorDynamics(
            transition_model=transition_model,size=dim_a,device=self.device,th_init=0.1*torch.randn(1)
            )
        self.trans_net = None
        
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)
        self.eta = None
        self.g_true=None
        self.internal_loss = internal_loss
        self.optMg_loss = optMg_loss
        self.coef_internal = coef_internal
        self.return_g = return_g

    def _encode_base(self, xs, enc):
        shape = xs.shape
        x = torch.reshape(xs, (shape[0] * shape[1], *shape[2:]))
        H = enc(x)
        H = torch.reshape(
            H, (shape[0], shape[1], *H.shape[1:]))
        return H

    def encode(self, xs):
        H = self._encode_base(xs, self.enc)
        H = torch.reshape(
            H, (H.shape[0], H.shape[1], self.dim_m, self.dim_a))
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(self.change_of_basis,
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        return H

    def phi(self, xs):
        return self._encode_base(xs, self.enc.phi)

    def get_M(self, xs):
        dyn_fn = self.dynamics_fn(xs)
        return dyn_fn.M
    
    def set_eta(self, eta):
        self.eta = eta

    def decode(self, H):
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(torch.linalg.inv(self.change_of_basis),
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        n, t = H.shape[:2]
        if hasattr(self, "pidec"):
            H = rearrange(H, 'n t d_s d_a -> (n t) d_a d_s')
            H = self.pidec(H)
        else:
            H = rearrange(H, 'n t d_s d_a -> (n t) (d_s d_a)')
        x_next_preds = self.dec(H)
        x_next_preds = torch.reshape(
            x_next_preds, (n, t, *x_next_preds.shape[1:]))
        return x_next_preds

    def dynamics_fn(self, xs, return_loss=False, fix_indices=None):
        H = self.encode(xs)
        return self.dynamics_model(H, return_loss=return_loss, fix_indices=fix_indices, eta=self.eta)


    def switch_transition_model(self, second_transition):
        self.transition_model = second_transition
        self.dynamics_model = LinearTensorDynamics(
            transition_model=second_transition,size=self.dim_a,device=self.device
            )


    def loss(self, xs, gelement=None, return_reg_loss=True, T_cond=2, reconst=False,return_g=False):

        xs_cond = xs[:, :T_cond]
        # KF: main call of the model
        if self.transition_model=='Fixed':
            self.g_true=gelement

        # reconstuction and prediction    
        xs_pred = self(xs_cond, gelement=self.g_true,return_reg_loss=return_reg_loss,
                       n_rolls=xs.shape[1] - T_cond, predictive=self.predictive, reconst=reconst, return_g=return_g)
        
        if return_reg_loss:
            xs_pred, H_pred, reg_losses = xs_pred

        if reconst:
            xs_target = xs
            xs_target_recon = xs[:,:T_cond,:]      # 0:T_cond includes the reconstruction, while T_cond: is the prediction
            xs_target_pred = xs[:,T_cond:,:]
        else:
            xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]    # prediction only
        ndim = xs_target.ndim     
        # standard loss for the prediction and/or reconstruction  
        loss = torch.mean(
            torch.sum((xs_target - xs_pred) ** 2, axis=tuple(range(2,ndim)))
            )  

        
        if self.internal_loss:    # add latent loss to the returned loss
            loss = loss + 0.01*reg_losses[2]     # add internal_loss_T  # mean squared error all batch data 


        loss_reconst = torch.mean(
            torch.sum((xs_target_recon - xs_pred[:,:T_cond,:]) ** 2, axis=tuple(range(2,ndim)))
            )

        reg_losses = reg_losses + (loss_reconst,)
        loss_pred = torch.mean(
            torch.sum((xs_target_pred - xs_pred[:,T_cond:,:]) ** 2, axis=tuple(range(2,ndim)))
            )
        reg_losses = reg_losses + (loss_pred,)

        return (loss, reg_losses) if return_reg_loss else loss
        

    def __call__(
            self, 
            xs_cond, 
            return_reg_loss=False, 
            n_rolls=1, 
            fix_indices=None, 
            predictive=True, 
            reconst=False, 
            gelement=None,       
            fixedM=False,
            return_g=False
            ):
        # Encoded Latent. Num_ts x len_ts x  dim_m x 
        H = self.encode(xs_cond)

        if self.transition_model=='Fixed':
            if gelement is not None:
                self.g_true=gelement

        # ==Esitmate dynamics==
        ret = self.dynamics_model(
            H, gelement=gelement, return_loss=return_reg_loss, fix_indices=fix_indices, eta=self.eta, return_g=return_g, net_g=self.trans_net)
        if return_reg_loss:
            # fn is a map by M_star. Loss is the training external loss
            fn, losses = ret
        else:
            fn = ret

        if predictive:
            H_last = H[:, -1:]      # \Phi( x[:,T_cond-1]) last time point used for estimating M 
            H_preds = [H] if reconst else []    # If reconst, \Phi( x[:,0:T_cond]) ), used for estimating reconstuction error.
            array = np.arange(n_rolls)
        else:
            H_last = H[:, :1]           # \Phi( x[:,0] ), reconstruction error is used. 
            H_preds = [H[:, :1]] if reconst else []         # \Phi( x[:,0]) ) if reconst
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)   # size of H - 1

        for _ in array:
            H_last = fn(H_last)     # roll-out
            H_preds.append(H_last)  
        H_preds = torch.cat(H_preds, axis=1)
        # if predictive, H_preds contains \Phi(x[:,0:T_cond]) for the first part, and M^k \Phi(x[:,T_cond-1])) for the rest.
        # else, H_preds contains M^k \Phi(x[:,0]) in the time points used for estimating M (thus, re-using the data)        

        # Prediction in the observation space
        x_preds = self.decode(H_preds)

        if return_reg_loss:
            return x_preds, H_preds, losses
        else:
            return x_preds


    def testing(
        self, 
        xs, 
        T_cond=2,
        return_reg_loss=True, 
        n_rolls=1, 
        predictive=True, 
        reconst=False,    
        return_g=False, 
        gelement=None,
        ):
        # Encoded Latent. Num_ts x len_ts x  dim_m x 
        # gelement IS used in this function
        xs_cond = xs[:, :T_cond]

        if self.transition_model=='Fixed':
            gelement = gelement
        else:
            gelement = None
        #
        # training disabled for the model.
        for param in self.parameters():
            param.requires_grad = False

        H = self.encode(xs_cond)
        H0, H1 = H[:, :-1], H[:, 1:]

        # ==Esitmate dynamics==
        ret = self.dynamics_model.evaluation(H, return_loss=return_reg_loss, net_g=self.trans_net, return_g=return_g, gelement=gelement)
        fn = ret[0]
        if return_reg_loss:
            # fn is a map by M_star. Loss is the training external loss
            reg_losses = ret[1]

        if predictive:
            H_last = H[:, -1:]              # \Phi( x[:,T_cond-1]) last time point used for estimating M 
            H_preds = [H] if reconst else []    # If reconst, \Phi( x[:,0:T_cond]) ), used for estimating reconstuction error.
            array = np.arange(n_rolls)
        else:
            H_last = H[:, :1]               # \Phi( x[:,0] ), reconstruction error is used. 
            H_preds = [H[:, :1]] if reconst else []     # \Phi( x[:,0]) ) if reconst
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)   # size of H - 1

        for _ in array:
            H_last = fn(H_last)     # roll-out
            H_preds.append(H_last)
        H_preds = torch.cat(H_preds, axis=1)    
        # if predictive, H_preds contains \Phi(x[:,0:T_cond]) for the first part, and M^k \Phi(x[:,T_cond-1])) for the rest.
        # else, H_preds contains M^k \Phi(x[:,0]) in the time points used for estimating M (thus, re-using the data)

        # Prediction in the observation space
        x_preds = self.decode(H_preds)

        #
        # training is abled before quitting
        for param in self.parameters():
            param.requires_grad = True

        if reconst:
            xs_target = xs
        else:
            xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]
        ndim = xs_target.ndim       
        loss = torch.mean(
            torch.sum((xs_target - x_preds) ** 2, axis=tuple(range(2,ndim)))    # mean squared error all batch data 
            )

        ret = (loss, x_preds)
        if return_reg_loss:
            ret = ret +  (reg_losses, )
        
        return ret


    def rollout(
        self, 
        xs, 
        T_cond=2,
        n_rolls=4, 
        predictive=True, 
        reconst=False,  
        return_g=False,  
        gelement=None,
        ):
        # Encoded Latent. Nbat x len_ts x  dim_m x dim_a
        xs_cond = xs[:, :T_cond]

        if self.transition_model=='Fixed':
            gelement = gelement
        else:
            gelement = None

        #
        # training disabled for MgDGD.  The parameter theta must be optimized before evaluation
        if self.transition_model=='MgDGD':
            for param in self.parameters():
                param.requires_grad = False
            self.dynamics_model.transition.theta.requires_grad = False

        H = self.encode(xs_cond)

        # ==Esitmate dynamics==
        ret = self.dynamics_model.evaluation(H, return_loss=False, net_g=self.trans_net, return_g=return_g, gelement=gelement)
        fn = ret[0]

        if predictive:
            H_last = H[:, -1:]              # \Phi( x[:,T_cond-1]) last time point used for estimating M 
            H_preds = [H] if reconst else []    # If reconst, \Phi( x[:,0:T_cond]) ), used for estimating reconstuction error.
            array = np.arange(n_rolls)
        else:
            H_last = H[:, :1]               # \Phi( x[:,0] ), reconstruction error is used. 
            H_preds = [H[:, :1]] if reconst else []     # \Phi( x[:,0]) ) if reconst
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)   # size of H - 1

        # roll-out fn
        for _ in array:
            H_last = fn(H_last)     
            H_preds.append(H_last)
        H_preds = torch.cat(H_preds, axis=1)    
        # if predictive, H_preds contains \Phi(x[:,0:T_cond]) for the first part, and M^k \Phi(x[:,T_cond-1])) for the rest.
        # else, H_preds contains M^k \Phi(x[:,0]) in the time points used for estimating M (thus, re-using the data)

        # Prediction in the observation space
        x_preds = self.decode(H_preds)

        #
        # training is abled before quitting for MgDGD
        if self.transition_model=='MgDGD':
            for param in self.parameters():
                param.requires_grad = True
            self.dynamics_model.transition.theta.requires_grad = True

        return x_preds

    

    # plot rollout
    def plot_rollout(self, fpath, x_preds, x_true, x, col_true='b-', col_est='r-'):
        import matplotlib.pyplot as plt

        Nfig_h = x_preds.shape[0]   
        Nfig_v = 2

        f_preds=x_preds.cpu()
        f_preds=f_preds.clone().detach().numpy()
        f_true=x_true.cpu()
        f_true=f_true.clone().detach().numpy()

        plt.style.use('ggplot')
                
        fig = plt.figure(figsize=[18, 5])
        for i in range(f_preds.shape[0]):
            ax = fig.add_subplot(Nfig_v, Nfig_h, i+1)
            ax.plot(x, f_preds[i,:], col_est)
            plt.xlabel('time', fontsize=16)
            plt.ylabel('values', fontsize=16)
            plt.title('Roll-out: t={}'.format(i)) 

        for i in range(f_preds.shape[0],f_preds.shape[0]+f_true.shape[0]):
            ax = fig.add_subplot(Nfig_v, Nfig_h, i+1)
            ax.plot(x, f_true[i-f_preds.shape[0],:], col_true)
            plt.xlabel('time', fontsize=16)
            plt.ylabel('values', fontsize=16)
            plt.title('Ground truth: t={}'.format(i-f_preds.shape[0])) 
    
        plt.plot()
        plt.tight_layout()         
        plt.savefig(fpath)
        plt.clf()
        plt.close()
        


        
    def loss_equiv(self, xs, T_cond=2, reduce=False):
        bsize = len(xs)
        xs_cond = xs[:, :T_cond]
        xs_target = xs[:, T_cond:]
        H = self.encode(xs_cond[:, -1:])
        dyn_fn = self.dynamics_fn(xs_cond)
        
        H_last = H
        H_preds = []
        n_rolls = xs.shape[1] - T_cond
        for _ in np.arange(n_rolls):
            H_last = dyn_fn(H_last)
            H_preds.append(H_last)
        H_pred = torch.cat(H_preds, axis=1)
        # swapping M
        dyn_fn.M = dyn_fn.M[torch.arange(-1, bsize-1)]
        
        H_last = H
        H_preds_perm = []
        for _ in np.arange(n_rolls):
            H_last = dyn_fn(H_last)
            H_preds_perm.append(H_last)
        H_pred_perm = torch.cat(H_preds_perm, axis=1)
        
        xs_pred = self.decode(H_pred)
        xs_pred_perm = self.decode(H_pred_perm)
        reduce_dim = (1,2,3,4,5) if reduce else (2,3,4)
        loss = torch.sum((xs_target-xs_pred)**2, dim=reduce_dim).detach().cpu().numpy()
        loss_perm = torch.sum((xs_target-xs_pred_perm)**2, dim=reduce_dim).detach().cpu().numpy()
        return loss, loss_perm
        




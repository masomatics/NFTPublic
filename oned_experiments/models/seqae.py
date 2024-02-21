import numpy as np
import torch
import torch.nn as nn
#from models import dynamics_models
import torch.nn.utils.parametrize as P
from models.dynamics_models import LinearTensorDynamics
from models.base_networks import MLPEncoder, MLPDecoder
import pdb

from einops import rearrange, repeat

'''
The model used in the experiment. Depends on the sub-model configuration files 
that determines the encoder-decoder architecture (base_networks.py)
and the regression in the linear latent space(dynamics_models.py). 
'''

class SeqAETSQmlp(nn.Module):
    def __init__(
            self,
            dim_a,
            dim_m,
            transition_model='LS',
            dim_data=128,
            alignment=False,
            change_of_basis=False,
            predictive=True,            
            gpu_id=0,
            activation='tanh'
            ):
        super().__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        if torch.cuda.is_available():
            self.device = torch.device("cuda",gpu_id) 
        else:
            self.device = torch.device("cpu")
        if activation == 'relu':
            activation_fxn = nn.ReLU()
        elif activation == 'tanh':
            activation_fxn = nn.Tanh()
        elif activation == 'sigmoid':
            activation_fxn = nn.Sigmoid()
        else:
            raise NotImplementedError

        self.enc = MLPEncoder(dim_data=dim_data, dim_latent=dim_a*dim_m, act=activation_fxn).to(self.device)
        self.dec = MLPDecoder(dim_latent=dim_a*dim_m,dim_data=dim_data, act=activation_fxn).to(self.device)
        self.transition_model = transition_model
        self.dynamics_model = LinearTensorDynamics(
            transition_model=transition_model,size=dim_a,device=self.device,th_init=0.1*torch.randn(1)
            )
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)
        self.eta = None
        self.g_true=None

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

    def loss(self, xs, gelement=None, return_reg_loss=True, T_cond=2, reconst=False, indices=None):
        xs_cond = xs[:, :T_cond]
        if self.transition_model=='Fixed':
            self.g_true=gelement

        # reconstuction and prediction    
        xs_pred = self(xs_cond, gelement=self.g_true,return_reg_loss=return_reg_loss,
                       n_rolls=xs.shape[1] - T_cond, predictive=self.predictive, reconst=reconst, indices=indices)
        
        if return_reg_loss:
            xs_pred, reg_losses = xs_pred

        if reconst:
            xs_target = xs
            xs_target_recon = xs[:,T_cond:,:]      # 0:T_cond includes the reconstruction, while T_cond: is the prediction
            xs_target_pred = xs[:,:T_cond,:]
        else:
            xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]    # prediction only
        ndim = xs_target.ndim       
        loss = torch.mean(
            torch.sum((xs_target - xs_pred) ** 2, axis=tuple(range(2,ndim)))
            )    # mean squared error all batch data 
        if reconst:
            loss_reconst = torch.mean(
                torch.sum((xs_target_recon - xs_pred[:,T_cond:,:]) ** 2, axis=tuple(range(2,ndim)))
                )
            loss_pred = torch.mean(
            torch.sum((xs_target_pred - xs_pred[:,:T_cond,:]) ** 2, axis=tuple(range(2,ndim)))
            )
        else:
            loss_reconst = 0
            loss_pred =  0
        #reg_losses = reg_losses + (loss_reconst,)
        reg_losses['loss_reconst'] = loss_reconst
        #reg_losses = reg_losses + (loss_pred,)
        reg_losses['loss_pred'] = loss_pred


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
            indices=None
            ):
        # Encoded Latent. Num_ts x len_ts x  dim_m x 
        # gelement is not used in this function
        H = self.encode(xs_cond)

        if self.transition_model=='Fixed':
            self.g_true=gelement

        # ==Esitmate dynamics==
        ret = self.dynamics_model(
            H, gelement=self.g_true, return_loss=return_reg_loss, fix_indices=fix_indices, eta=self.eta, indices=indices)
        if return_reg_loss:
            # fn is a map by M_star. Loss is the training external loss
            fn, losses = ret
        else:
            fn = ret

        if predictive:
            H_last = H[:, -1:]      # intial state for the recursive applications of fn
            H_preds = [H] if reconst else []    # if reconst, H_preds[:,0:1,:,:] contains the encoder output for xs_cond[:,0:2,:]
            array = np.arange(n_rolls)
        else:
            H_last = H[:, :1]
            H_preds = [H[:, :1]] if reconst else []
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)

        for _ in array:
            H_last = fn(H_last)     # roll-out
            H_preds.append(H_last)  # if reconst, H_preds[:,0:2,:,:] contains encoder outputs, while H_preds[:,2:,:,:] is the recursive outputs of predictor fn
        H_preds = torch.cat(H_preds, axis=1)
        # Prediction in the observation space
        x_preds = self.decode(H_preds)

        if return_reg_loss:
            return x_preds, losses
        else:
            return x_preds


    def testing(
        self, 
        xs, 
        T_cond=2,
        return_reg_loss=False, 
        n_rolls=1, 
        predictive=True, 
        reconst=False,     
        ):
        # Encoded Latent. Num_ts x len_ts x  dim_m x 
        # gelement IS used in this function
        xs_cond = xs[:, :T_cond]

        #
        # training disabled for the model.
        for param in self.parameters():
            param.requires_grad = False

        H = self.encode(xs_cond)

        # ==Esitmate dynamics==
        ret = self.dynamics_model.evaluation(H, return_loss=return_reg_loss)
        if return_reg_loss:
            # fn is a map by M_star. Loss is the training external loss
            fn, reg_losses = ret
        else:
            fn = ret

        if predictive:
            H_last = H[:, -1:]
            H_preds = [H] if reconst else []
            array = np.arange(n_rolls)
        else:
            H_last = H[:, :1]
            H_preds = [H[:, :1]] if reconst else []
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)

        for _ in array:
            H_last = fn(H_last)
            H_preds.append(H_last)
        H_preds = torch.cat(H_preds, axis=1)
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

        if return_reg_loss:
            return loss, x_preds, reg_losses
        else:
            return loss, x_preds
       
        



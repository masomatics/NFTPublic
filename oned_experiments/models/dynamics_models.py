import numpy as np
import torch
import torch.nn as nn
from utils.laplacian import make_identity_like, tracenorm_of_normalized_laplacian, make_identity, make_diagonal
import einops
import pytorch_pfn_extras as ppe
#from torchmin import minimize
import pdb
from base_networks import MLP 
import sys
sys.path.append('../')



def _rep_M(M, T):
    return einops.repeat(M, "n a1 a2 -> n t a1 a2", t=T)


def _loss(A, B):
    return torch.sum((A-B)**2)


def _solve(A, B):
    ATA = A.transpose(-2, -1) @ A
    ATB = A.transpose(-2, -1) @ B
    return torch.linalg.solve(ATA, ATB)


def loss_bd(M_star, alignment):
    # Block Diagonalization Loss
    S = torch.abs(M_star)
    STS = torch.matmul(S.transpose(-2, -1), S)
    if alignment:
        laploss_sts = tracenorm_of_normalized_laplacian(
            torch.mean(STS, 0))
    else:
        laploss_sts = torch.mean(
            tracenorm_of_normalized_laplacian(STS), 0)
    return laploss_sts



def loss_orth(M_star):
    # Orthogonalization of M
    I = make_identity_like(M_star)
    return torch.mean(torch.sum((I-M_star @ M_star.transpose(-2, -1))**2, axis=(-2, -1)))

    

class LinearTensorDynamics(nn.Module):

    class DynFn(nn.Module):
        def __init__(self, M):
            super().__init__()
            self.M = M

        def __call__(self, H):
            return H @ _rep_M(self.M, T=H.shape[1])

        def inverse(self, H):
            M = _rep_M(self.M, T=H.shape[1])
            return torch.linalg.solve(M, H.transpose(-2, -1)).transpose(-2, -1)
        
        def loss(self,H0,H1):
            dif = H0 @ _rep_M(self.M, T=H0.shape[1]) - H1
            loss = torch.mean(
                torch.sum(dif ** 2, axis=tuple(range(2,H0.ndim)))    # mean squared error all batch data 
            )
            return loss
                
    class TransLS():
        def __init__(self, fix_indices):
            self.fix_indices=fix_indices
        
        def __call__(self, H0, H1, indices='None'):
            _H0 = H0.reshape(H0.shape[0], -1, H0.shape[-1])
            _H1 = H1.reshape(H1.shape[0], -1, H1.shape[-1])
            if self.fix_indices is not None:
                # Note: backpropagation is disabled.
                dim_a = _H0.shape[-1]
                active_indices = np.array(list(set(np.arange(dim_a)) - set(self.fix_indices)))
                _M_star = _solve(_H0[:, :, active_indices],
                                _H1[:, :, active_indices])
                M_star = make_identity(_H1.shape[0], _H1.shape[-1], _H1.device)
                M_star[:, active_indices[:, np.newaxis], active_indices] = _M_star
            else:
                M_star = _solve(_H0, _H1)
            return M_star
                
    def __init__(self, transition_model, fix_indices=None, th_init=None, size=None, device=None, eta=None, alignment=True, datsize=None, basenetargs={}, dim_m=128):
        super().__init__()
        self.fix_indices=fix_indices
        self.th_init=th_init
        self.device=device
        self.size=size
        self.transition_model = transition_model
        self.eta = eta
        self.alignment = alignment
        self.datsize=datsize
        self.dim_m = dim_m
        if transition_model=='LS':      # either of LS, Fixed, MgDGD
            self.transition=self.TransLS(fix_indices=self.fix_indices)
        else:
            print('Error: transition_model must be either of LS, Fixed, or MgDGD')
            

    def __call__(self, H, gelement=None, return_loss=False, fix_indices=None, eta=None, indices=None):
        # Regress M.
        # Note: backpropagation is disabled when fix_indices is not None.
        # torch tensors are used. 

        self.fix_indices = fix_indices
        self.eta = eta
        self.g_true=gelement
        # H0.shape = H1.shape [n, t, s, a]
        H0, H1 = H[:, :-1], H[:, 1:]
        # num_ts x ([len_ts -1] * dim_s) x dim_a
        # The difference between the time shifted components
        loss_internal_0 = _loss(H0, H1)
        ppe.reporting.report({
            'loss_internal_0': loss_internal_0.item()
        })

        if self.transition_model == 'MgDGD':
            self.transition.set_eta(self.eta)
        elif self.transition_model == 'Fixed':
            self.transition.set_gtrue(self.g_true)

        M_star = self.transition(H0,H1, indices=indices)
                           
        dyn_fn = self.DynFn(M_star)
        loss_internal_T = _loss(dyn_fn(H0), H1)
        ppe.reporting.report({
            'loss_internal_T': loss_internal_T.item()
        })

        # M_star is returned in the form of module, not the matrix
        if return_loss:
            # losses = (loss_bd(dyn_fn.M, self.alignment),
            #           loss_orth(dyn_fn.M), loss_internal_T)
            losses = {'loss_bd': loss_bd(dyn_fn.M, self.alignment),
            'loss_orth': loss_orth(dyn_fn.M), 
            'loss_internal': loss_internal_T}
            return dyn_fn, losses
        else:
            return dyn_fn


    def evaluation(self, H, return_loss):
        H0, H1 = H[:, :-1], H[:, 1:]

        if self.transition_model=='LS':      # either of LS, Fixed, MgDGD
            test_Mstar=self.transition(H0,H1)
        else:
            raise NotImplementedError

        dyn_fn = self.DynFn(test_Mstar)

        loss_internal_T = _loss(dyn_fn(H0), H1)
        if return_loss:
            losses = (loss_bd(dyn_fn.M, self.alignment),
                      loss_orth(dyn_fn.M), loss_internal_T)
            return dyn_fn, losses
        else:
            return dyn_fn

#########################################################################################################
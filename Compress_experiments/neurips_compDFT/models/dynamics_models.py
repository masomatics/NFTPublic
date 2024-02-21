import numpy as np
import torch
import torch.nn as nn
from utils.laplacian import make_identity_like, tracenorm_of_normalized_laplacian, make_identity, make_diagonal
import einops
import pytorch_pfn_extras as ppe



def _rep_M(M, T):
    return einops.repeat(M, "n a1 a2 -> n t a1 a2", t=T)

def _rep2_M(M, B, T):
    return einops.repeat(M, "a1 a2 -> n t a1 a2", n=B, t=T)


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


class loss_lse():
    def __init__(self,H0,H1,device):
        self.H0 = H0
        self.H1 = H1
        self.size = H0.shape[2]
        self.device = device
    def __call__(self,angle):
        M = Mg_rot2D_func(angle,self.size).to(self.device)
        A = self.H0 @ M - self.H1
        return torch.sum(torch.square(A))


class Mg_rot2D():
    def __init__(self,size,device):
        self.size = size
        self.k = int(np.floor(size/2))
        self.mask = torch.kron(torch.eye(self.k),torch.ones(2,2)).to(device)   # [[1,1],[1,1]] block diagonal matrix of size k
        self.mult = torch.arange(self.k).to(device)

    def __call__(self, angle, size):
        nbat = angle.shape[0]
        angle = angle.reshape(nbat,1)
        mask_bat = self.mask.repeat(nbat,1,1)
        mult_bat = self.mult.repeat(nbat,1)     # [nbat x k] (0, 1, ..., k-1)
        th = angle.repeat(1,self.k) * mult_bat  # [nbat, k] matrix containing  theta(a) * j for (a,j,1)
        Mn = [torch.cos(th),-torch.sin(th),torch.sin(th),torch.cos(th)]
        Mn = torch.stack(Mn,2)        # Mn: [nbat, k, 4]
        Mn = Mn.reshape(nbat,2*self.k,2)  # the rotation matrices (2x2) are stacked 
        M = Mn.repeat(1,1,self.k)
        M = (M * mask_bat).to(torch.float32)
        return M

    

# new main dynamics function by KF
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

    class DynFnMg(nn.Module):
        def __init__(self, nbat, size, device, th_ini=None):
            super().__init__()
            if th_ini is not None:
                if nbat == len(th_ini):
                    self.theta = nn.Parameter(th_ini.to(device))
                else:
                    print('Error: the size of initial theta must match')
            else:
                self.theta = nn.Parameter(torch.randn(nbat).to(device))
            self.nbat = nbat
            self.size = size
            self.device = device

        def __call__(self, H):
            Mg = Mg_rot2D(self.size, self.device)
            M = Mg(self.theta,self.size).to(self.device)
            return H @ _rep_M(M, T=H.shape[1])

        def loss(self,H0,H1):
            dif = self(H0) - H1
            loss = torch.mean(
                torch.sum(dif ** 2, axis=tuple(range(2,H0.ndim)))    # mean squared error all batch data 
            )
            return loss

        def get_g(self):
            return self.theta
            

        
    class TransLS():
        def __init__(self, fix_indices):
            self.fix_indices=fix_indices
        
        def __call__(self, H0, H1):
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


    class TransFixed():
        def __init__(self, size, device):
            self.size = size
            self.device = device
            self.Mg = Mg_rot2D(size,device)   
            self.g_true = None    

        def __call__(self, H0, H1):     # H0 and H1 are not used in this class
            M_star = self.Mg(self.g_true,self.size).to(self.device)     # H.shape[3]=dim_a (matrix dim)  
            return M_star
        
        def set_gtrue(self,gelement):
            self.g_true = gelement
        

    class AbelMSP():
        def __init__(self, size, normalize=False):
            self.size = size
            self.k = size//2
            self.mask = torch.kron(torch.eye(self.k), torch.ones(2, 2))
            self.normalize = normalize # if true the estimated matrix are normalized so that each block will be an SO(2) mat
    
        def complex_lsq(self, x, y):
            """
            Args:
                x: Tensor of shape [...,K,freqs,2]
                y: Tensor of shape [...,K,freqs,2]
            Retrun:
                complex least square solution of shape [...,freqs, 2]
            """
            sq = torch.sum(y**2, dim=-1)
            sq = sq + 1e-12*(sq == 0)
            sq_sum = torch.sum(sq, dim=-2)
            real = (x[..., 0] * y[..., 0] +  x[..., 1] * y[..., 1]).sum(-2)
            img = (x[..., 1] * y[..., 0] - x[..., 0] * y[..., 1]).sum(-2)
            return torch.stack([real/sq_sum, img/sq_sum], -1)

        def __call__(self, H0, H1):
            batch_size, freqs = H0.shape[0], H0.shape[-1]//2
            assert freqs == self.k
            H0 = H0.reshape(batch_size, -1, H0.shape[-1]//2, 2) #[B, K, Freqs, 2] 
            H1 = H1.reshape(batch_size, -1, H1.shape[-1]//2, 2) #[B, K, Freqs, 2]
            diagM = self.complex_lsq(H1, H0) # [B, Freqs, 2]

            ### Return real 2x2 block-matrix
            if self.normalize:
                diagM = torch.nn.functional.normalize(diagM, dim=-1)
            M = [diagM[..., 0], -diagM[..., 1], diagM[..., 1], diagM[..., 0]] 
            M = torch.stack(M, -1) # [B, Freqs, 4]
            M = M.reshape(batch_size, 2*freqs, 2) #  [B, 2*Freqs, 2]
            M = M.repeat(1, 1, freqs) #  [B, K, 2*Freqs, 2*Freqs]
            mask_bat = self.mask.to(H0.device)[None].repeat(batch_size, 1, 1)
            M = mask_bat * M
            return M

    
        
    def __init__(self, transition_model, fix_indices=None, th_init=None, size=None, device=None, eta=None, alignment=True):
        super().__init__()
        self.fix_indices=fix_indices
        self.th_init=th_init
        self.device=device
        self.size=size
        self.transition_model = transition_model
        self.eta = eta
        self.alignment = alignment
        if transition_model=='LS':      # either of LS, Fixed, MgDGD
            self.transition=self.TransLS(fix_indices=self.fix_indices)
        elif transition_model=='Fixed':
            self.transition=self.TransFixed(size=self.size, device=self.device)
        elif transition_model == 'AbelMSP':      
            self.transition = self.AbelMSP(self.size)
        else:
            print('Error: transition_model must be either of LS, Fixed, or AbelMSP.')
            

    def __call__(self, H, gelement=None, return_loss=True, fix_indices=None, eta=None, return_g=False, net_g=None):
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

        if self.transition_model == 'Fixed':
            self.transition.set_gtrue(self.g_true)

        # dyn_fn: M is computed depending on transition_model
        # M_star is returned in the form of module, not the matrix
        # M_star = self.transition(H0,H1)   
        g_est=[]
        if self.transition_model in ['LS', 'Fixed', 'AbelMSP']:      # either of LS, Fixed, MgDGD
            M_star=self.transition(H0,H1)
                              
        dyn_fn = self.DynFn(M_star)
        loss_internal_T = _loss(dyn_fn(H0), H1)

        ppe.reporting.report({
            'loss_internal_T': loss_internal_T.item()
        })

        if return_loss:
            losses = (loss_bd(dyn_fn.M, self.alignment),
                      loss_orth(dyn_fn.M), loss_internal_T)
            return dyn_fn, losses
        else:
            return dyn_fn


    def evaluation(self, H, return_loss, gelement=None, net_g=None, return_g=False):
        H0, H1 = H[:, :-1], H[:, 1:]

        if self.transition_model == 'Fixed':
            self.transition.set_gtrue(gelement)

        if self.transition_model in ['LS', 'AbelMSP']:      # either of LS, Fixed, MgDGD
            test_Mstar=self.transition(H0,H1)
        elif self.transition_model=='Fixed':
            test_Mstar=self.transition(H0,H1)       
        else:
            print('Error: transition_model must be either of LS, Fixed, AbelMSP')

        # fix M(g)
        dyn_fn = self.DynFn(test_Mstar)

        dif = dyn_fn(H0) - H1
        loss_internal_T = torch.mean(
                torch.sum(dif ** 2, axis=tuple(range(2,H0.ndim)))    # mean squared error all batch data 
            )
        ret = (dyn_fn, )
        if return_loss:
            ret = ret + (loss_internal_T,)

        return ret
        

            

class LinearTensorDynamicsLSTSQ(nn.Module):

    class DynFn(nn.Module):
        def __init__(self, M):
            super().__init__()
            self.M = M

        def __call__(self, H):
            return H @ _rep_M(self.M, T=H.shape[1])

        def inverse(self, H):
            M = _rep_M(self.M, T=H.shape[1])
            return torch.linalg.solve(M, H.transpose(-2, -1)).transpose(-2, -1)

    def __init__(self, alignment=True):
        super().__init__()
        self.alignment = alignment

    def __call__(self, H, return_loss=False, fix_indices=None):
        # Regress M.
        # Note: backpropagation is disabled when fix_indices is not None.
        # torch tensors are used. 

        # H0.shape = H1.shape [n, t, s, a]
        H0, H1 = H[:, :-1], H[:, 1:]
        # num_ts x ([len_ts -1] * dim_s) x dim_a
        # The difference between the time shifted components
        loss_internal_0 = _loss(H0, H1)
        ppe.reporting.report({
            'loss_internal_0': loss_internal_0.item()
        })
        _H0 = H0.reshape(H0.shape[0], -1, H0.shape[-1])
        _H1 = H1.reshape(H1.shape[0], -1, H1.shape[-1])
        if fix_indices is not None:
            # Note: backpropagation is disabled.
            dim_a = _H0.shape[-1]
            active_indices = np.array(list(set(np.arange(dim_a)) - set(fix_indices)))
            _M_star = _solve(_H0[:, :, active_indices],
                             _H1[:, :, active_indices])
            M_star = make_identity(_H1.shape[0], _H1.shape[-1], _H1.device)
            M_star[:, active_indices[:, np.newaxis], active_indices] = _M_star
        else:
            M_star = _solve(_H0, _H1)
        dyn_fn = self.DynFn(M_star)
        loss_internal_T = _loss(dyn_fn(H0), H1)
        ppe.reporting.report({
            'loss_internal_T': loss_internal_T.item()
        })

        # M_star is returned in the form of module, not the matrix
        if return_loss:
            losses = (loss_bd(dyn_fn.M, self.alignment),
                      loss_orth(dyn_fn.M), loss_internal_T)
            return dyn_fn, losses
        else:
            return dyn_fn



'''
# fixed M(g)
class LinearTensorDynamicsfixedMTSQ(nn.Module):

    class DynFn(nn.Module):
        def __init__(self, M):
            super().__init__()
            self.M = M

        def __call__(self, H):
#            Mk = self.M.torch()
            Me = self.M
            return H @ _rep_M(Me, T=H.shape[1])

        def inverse(self, H):
            M = _rep_M(self.M, T=H.shape[1])
            return torch.linalg.solve(M, H.transpose(-2, -1)).transpose(-2, -1)
        
    def get_g(self):
        return self.g_est

    def __init__(self,alignment,opttheta=False,searchmethod=None,device=None):
        super().__init__()
        self.fixedM = True
        self.alignment = alignment
        self.Mg = Mg_rot2D(self.fixedM)
        self.opttheta = opttheta
        self.device = device
        self.searchmethod = TotalSearchTh(self.device)
        self.g_est = None

    def __call__(self, H, device, gelement, return_loss=False, fix_indices=None, fixedM = True):
        # return M(g) as Class DynFn by shift data. (If DynFn is called with argument H, then M(g) H is computed.)

        # H0.shape = H1.shape [n, t, s, a]
        H0, H1 = H[:, :-1], H[:, 1:]
        # num_ts x ([len_ts -1] * dim_s) x dim_a
        # The difference between the the time shifted components
        loss_internal_0 = _loss(H0, H1)
        ppe.reporting.report({
            'loss_internal_0': loss_internal_0.item()
        })

        if self.opttheta:
            angle=self.searchmethod(H0,H1,torch.zeros(H0.shape[0]))
        else:
            angle=gelement

        self.g_est = angle
        
        self.M = self.Mg(angle, size=H.shape[3]).to(device)     # H.shape[3]=dim_a (matrix dim)  
        # self.M contains nbat x dim_a x dim_a matrix 
        dyn_fn = self.DynFn(self.M).to(device)
        loss_internal_T = _loss(dyn_fn(H0), H1)
        ppe.reporting.report({
            'loss_internal_T': loss_internal_T.item()
        })

        # M * H is returned in the form of module, not the matrix
        if return_loss:
            losses = (loss_bd(dyn_fn.M, self.alignment),
                      loss_orth(dyn_fn.M), loss_internal_T)
            return dyn_fn, losses
        else:
            return dyn_fn
        
        


def Mg_rot2D_func(angle, size):
    k = int(np.floor(size/2))
    Mn = torch.eye(size)
    for i in range(k):
        th = i*angle
        Mn[2*i:2*i+2,2*i:2*i+2] = torch.tensor([[torch.cos(th), -torch.sin(th)], [torch.sin(th), torch.cos(th)]])
    M = Mn.to(torch.float32)
    return M

'''
class Mg_rot2Dfn(nn.Module):        # nn.Module M(g) for 2D rotation with g learnable parameter
    def __init__(self,Ndata,thnn,device=None):
        super(Mg_rot2Dfn, self).__init__()
        self.thnn = thnn
        # self.theta is a trainale parameter (Ndata dimension).  It should be initialized when the object is initialized
        self.device = device

    def __call__(self,dataidx,size):    # the resulting tensor is batchsize x dim_a x dim_a, where dim_a x dim_a contains R(m*th)
        k = int(np.floor(size/2))   
        nbat = dataidx.shape[0]
        th0 = self.thnn(dataidx)     # theta's for the current batch
        mult = torch.arange(k).repeat(nbat,1)
        th = torch.transpose(th0.squeeze().repeat(k,1),0,1) * mult.to(self.device)  # [nbat, k] matrix containing  th_i * j for (i,j)
        Mn = [torch.cos(th),-torch.sin(th),torch.sin(th),torch.cos(th)]
        Mn = torch.stack(Mn,2)        # Mn: [nbat, k, 4]
        Mn = Mn.reshape(nbat,2*k,2)  # for each i in [nbat], the rotation matrices (2x2) are stacked 
        M = Mn.repeat(1,1,k)
        mask = torch.kron(torch.eye(k),torch.ones(2,2)).repeat(nbat,1,1)   # nbat repetitions of [[1,1],[1,1]] block diagonal matrix of size k
        M = M * mask.to(self.device)
        return M.to(torch.float32)


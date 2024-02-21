import numpy
import torch
import torch.nn as nn
import copy 
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


'''
Code required for Character analysis.
'''

class Rotmat(nn.Module):
    def __init__(self, freqs):
        super().__init__()
        self.freqs = freqs
        self.num_blocks = len(freqs)
        
    def __call__(self, theta):
        mymat = torch.eye(len(self.freqs)*2)
        for k in range(self.num_blocks):
            mymat[2*k:2*(k+1),:][:, 2*k:2*(k+1)] = self.submat(theta, self.freqs[k])
        return mymat
        
    def submat(self, theta, freq):
        angle = theta * freq
        return torch.tensor([[np.cos(freq*theta), - np.sin(freq*theta)],
                     [np.sin(freq*theta), np.cos(freq*theta)]])


def inner_prod(rholist, gs, maxfreq=20, bins=50):
    #when computing over real represnetations, 
    #note that conjugate will also be counted, doubling the result.
    character_prod = []
    targfreqs = np.linspace(0,maxfreq,bins)

    for targfreq in tqdm(targfreqs):
        targobj = Rotmat([targfreq])

        inner_prod_vals = []
        for i in range(len(gs)):
            inner_prod_val = torch.trace(rholist[i]) * torch.trace(targobj(gs[i]))            
            inner_prod_vals.append(inner_prod_val)

        
        inner_prod_vals = torch.stack(inner_prod_vals)
        character_prod.append(torch.mean(inner_prod_vals).item())
        
    character_prod = torch.tensor(character_prod)
    return(targfreqs, character_prod)


def deltafxn(targfreq, freqs):
    output = 0 
    for k in range(len(freqs)):
        output = output + np.exp(-(freqs[k]-targfreq)**2 / 0.2 )
    return output 
        
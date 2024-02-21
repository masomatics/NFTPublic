import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
import pdb


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=3,
                 num_layer=3,
                 activation=nn.ELU,
                 hidden_multiple = 2,
                 initmode='default',
                 **kwargs):
        #super(MLP, self).__init__()
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        self.num_layer = num_layer
        self.dospec = dospec
        for k in range(num_layer):
            if k == 0:
                dimin = in_dim
                dimout = int(in_dim * hidden_multiple)
            else:
                dimin = int(in_dim * hidden_multiple)
                dimout = int(in_dim * hidden_multiple)
            linlayer = nn.Linear(in_features=dimin,
                                         out_features=dimout)
            if initmode == 'cond':
                initialize_linear(linlayer, thresh=2.0)
            elif initmode == 'default':
                nn.init.orthogonal_(linlayer.weight.data)
                nn.init.uniform_(linlayer.bias.data)
            else:
                raise NotImplementedError

            self.layers.append(linlayer)
            self.layers.append(activation())
        linlayer = nn.Linear(in_features=dimout,
                                         out_features=self.out_dim)


        self.layers.append(linlayer)

        self.network = nn.Sequential(*self.layers)

    def __call__(self, x):
        x = x.flatten(start_dim=1)
        return self.network(x)


# Four layer MLP for encoder
class MLPEncoder(nn.Module):
    def __init__(self,
                 dim_latent=128,
                 dim_hidden=256,
                 dim_data=128,
                 act=nn.ReLU(),
                 ):
        super().__init__()
        self.act = act
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.dim_data = dim_data
        self.lin1 = nn.Linear(self.dim_data,self.dim_hidden)
        self.lin2 = nn.Linear(self.dim_hidden,self.dim_latent)
        self.phi = nn.Sequential(
                # nn.LazyLinear(self.dim_hidden),
                self.lin1,
                self.act,
                # nn.LazyLinear(self.dim_hidden),
                self.lin2,
                self.act,
                # nn.LazyLinear(self.dim_latent)
                # nn.Linear(self.dim_hidden,self.dim_latent),
        )
        #self.linear = nn.LazyLinear(
        #    self.dim_latent) if self.dim_latent > 0 else lambda x: x  
        self.linear = nn.Linear(self.dim_latent,self.dim_latent)

    def __call__(self, x):
        h = x.to(torch.float32)
        h = self.phi(h)
        h = h.reshape(h.shape[0], -1)
        h = self.linear(h)
        return h


# Four layer MLP for encoder
class MLPDecoder(nn.Module):
    def __init__(self, 
                 dim_data = 128,
                 dim_latent = 128,
                 dim_hidden = 256,
                 act=nn.ReLU()               
           ):
        super().__init__()
        self.act = act
        self.dim_data = dim_data
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        # self.linear = nn.LazyLinear(self.dim_hidden)
        # self.linear = nn.Linear(self.dim_latent,self.dim_hidden)
        self.net = nn.Sequential(
            nn.Linear(self.dim_latent,self.dim_hidden),
            self.act,
            nn.Linear(self.dim_hidden,self.dim_hidden),
            self.act,
            #nn.LazyLinear(self.dim_data)
            nn.Linear(self.dim_hidden,self.dim_data)
        )

    def __call__(self, x):
        x = self.net(x)
        return x




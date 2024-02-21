# Data generation for neural fft
# Sequence of periodic functions on circle [0, 2\pi ) 
# shift is acted to a function 
# Modified from seq_mnist.py

import os
import numpy as np
import math
from copy import deepcopy



class Shifted_FreqFun():
    # data is generated at initialization
    def __init__(
            self,
            Ndata=5000,   # Number of data
            N=128,          # Number of observation points 
            T=3,            # Time steps in a sequence
            shift_label=False,
            indexing=False,
            fixedM=False,
            batchM_size=1,
            shift_range=2,  # frequency N/(5*shift_range)
            max_shift=[0.0, 2 * math.pi / 2],   # range of shift action
            max_T=5,
            shared_transition=False,
            rng=None,
            nfreq=5,        # Number of selected frequecy to make a function
            coef=None,      # Coefficients of the selected frequencies. 
            ns=0.0,          # Noise level of additive Gaussian noise
            shifts = None,
            random_shifts = False   # If true, each data has different shifts over epochs
    ):
        self.T = T             
        self.max_T = max_T     
        self.Ndata = Ndata
        self.N = N                    
        self.ns = ns            
        self.rng = rng if rng is not None else np.random
        self.nfreq = nfreq      
        self.shift_range = shift_range      # control the frequency range of data
        self.max_shift = max_shift    
        self.shift_label = shift_label
        self.fixedM = fixedM
        self.indexing = indexing
        self.batchM_size = batchM_size
        self.shifts = shifts
        self.random_shifts = random_shifts
        
        if coef is None:
            coef = np.random.randn(nfreq)
            coef = coef/np.linalg.norm(coef)
        self.coef = coef
        
        dt = 1.0/N
        t = np.arange(0, N*dt, dt) # time domain 
        
        # Generation of data (size: Ndata, dim: N)
        fdata =[]
        for i in range(Ndata):
            # Initial function data generated: f(t) = \sum_j coef_j * sin( 2 \pi freqsel_j t)
            freqsel = np.random.randint(0,np.ceil(N/(5*self.shift_range)),nfreq)  # randomly selected frequencies
            coef = np.random.randn(nfreq)

            f = np.matmul(np.sin(np.outer(2*np.pi*t,freqsel)), coef)
            fdata.append (f + ns * np.random.randn(N))
                       
        self.data = np.array(fdata)     # data:  Ndata x N (double array)      
        
        
        #   for fixedM senairo,the same M(g) is used in a batch.  Set shuffle off in Dataloader. 
        #   self.shifts specify the shift
        if self.fixedM and (self.shifts is None):
            shifts=[]
            for i in range(Ndata):
                if i % batchM_size == 0:
                    shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)
                shifts.append(shift)
            self.shifts=np.array(shifts)
        elif self.random_shifts is False:
            #  random shift (this fixes shift for each data)
            self.shifts = self.max_shift[0] + np.random.rand(Ndata) * (self.max_shift[1] - self.max_shift[0])
        else:
            self.shifts = np.zeros(Ndata)   # shifts are assigned in __getitem__ at dataloader


        self.shared_transition = shared_transition
        if self.shared_transition:
            self.init_shared_transition_parameters()


    def init_shared_transition_parameters(self):
        self.shared_shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, i):
        fun = self.data[i]  # i-th data (single function data (N-dim vector))
        
        if self.random_shifts:  # random shifts at every epoch
            shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)
            self.shifts[i] = shift
        elif self.shared_transition:
            shift = self.shared_shift 
        else:
            shift = self.shifts[i]          
            
        fvals = []
        for t in range(self.T):     
            sft_idx = int(np.floor(t*shift*self.N/(2*math.pi))) # time point corresponding to shift
            fn_t = np.roll(fun, sft_idx)     # fn_t(i) = f(i-sft_idx)  (mod N)
            fvals.append(fn_t)              # data values are the list of length T, each of which contains N dim functions values.

        retval = [fvals]
        retval.append(shift)
        if self.indexing:
            idx = np.zeros(self.Ndata)
            idx[i] = 1.0
            retval.append(idx)  # idx is a one-hot Ndata-dim vector with 1 at the index of the data

        return retval       # [data, gelement, [dataidx]] (indexing is added if self.indexing is True)


            

#  time points are nonlinearly transformed from the latent function, where the standard shifts are applied.  
class Shifted_FreqFun_nl():
    def __init__(
            self,
            Ndata=5000,   # Number of data
            N=128,          # Number of observation points 
            T=3,            # Time steps in a sequence
            shift_label=False,
            batchM_size=16,
            shift_range=2,
            max_shift=[0.0, 2 * math.pi / 2],   # range of shift action (in radian)
            max_T=5,
            shared_transition=False,
            rng=None,
            indexing=False,
            nfreq=16,        # Number of selected frequecy to make a function
            coef=None,      # Coefficients of the selected frequencies. 
            ns=0.0,          # Noise level of additive Gaussian noise
            pow=3,
            shifts=None,
            random_shifts=False
    ):
        self.T = T             
        self.max_T = max_T     
        self.Ndata = Ndata
        self.N = N                    
        self.ns = ns            
        self.rng = rng if rng is not None else np.random
        self.nfreq = nfreq      
        self.shift_range = shift_range      # control the frequency range of data
        self.max_shift = max_shift    
        self.shift_label = shift_label
        self.batchM_size = batchM_size
        self.shifts = shifts
        self.pow = pow
        
        if coef is None:
            coef = np.random.randn(nfreq)
            coef = coef/np.linalg.norm(coef)
            #coef = coef/nfreq
        self.coef = coef
        
        freqs = []
        for i in range(Ndata):
            freqsel = np.random.randint(0,np.ceil(N/(5*self.shift_range)),nfreq)  # randomly selected frequencies
            freqs.append (freqsel)
                      
   
        self.freqs = np.array(freqs)     # self.freqs:  Ndata x nfreq (double array) contains frequencoes to make the latent functions
        #  To make the function values use f =  np.matmul(np.sin(np.outer(2*np.pi*t,self.freqs[i,:])), self.coef)  # function value at latent t  
        
        #   In the batch, the same M(g) is used.  Set shuffle off in Dataloader. 
        if self.shift_label and (self.shifts is None):
            shifts=[]
            for i in range(Ndata):
                if i % batchM_size == 0:
                    shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)
                shifts.append(shift)
            self.shifts=np.array(shifts)        # shifts are given in radian [0, 2pi)

        self.shared_transition = shared_transition
        if self.shared_transition:
            self.init_shared_transition_parameters()


    def init_shared_transition_parameters(self):
        self.shared_shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)


    def __len__(self):
        return self.Ndata


    def __getitem__(self, i):       # i-th data and its shifts (T shifts)
        freq = self.freqs[i,:]
        dt = 1.0/self.N
        obs_t = np.arange(0, self.N*dt, dt) # observed time  [0,1]/N
        lat_t = np.power(obs_t, self.pow)      # observed time domain: lat_t = obs_t^pow
                
        if self.shift_label:
            shift = self.shifts[i]          
        elif self.shared_transition:
            shift = self.shared_shift 
        else:
            shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)
            
        fvals = []
        for t in range(self.T):     # shift * t ( action up to T times)
            lat_t = np.power(obs_t, self.pow) - t*shift/(2*math.pi)
            lat_t = lat_t + (lat_t<0)*(1.0)          
            fobs_t = np.matmul(np.sin(np.outer(2*np.pi*lat_t,freq)), self.coef)      # fn_t(j) = f_lat((t_j)**pow - shift*t)  
            fvals.append(fobs_t + self.ns * np.random.randn(self.N))

        if self.shift_label:
            return [fvals, shift]    # List (length T) of N-dim function values and shift 
        else:
            return fvals        # List (length T) of N-dim function values and time 
               


class Shifted_FixedFreqFun_nl():
    def __init__(
            self,
            Ndata=5000,   # Number of data
            N=128,          # Number of observation points 
            T=3,            # Time steps in a sequence
            shift_label=False,
            batchM_size=16,
            shift_range=2,
            max_shift=[0.0, 2 * math.pi / 2],   # range of shift action (in radian)
            max_T=5,
            shared_transition=False,
            rng=None,
            indexing=False,
            nfreq=5,        # Number of selected frequecy to make a function
            coef=None,      # Coefficients of the selected frequencies. 
            ns=0.0,          # Noise level of additive Gaussian noise
            pow=3,
            shifts=None,
            random_shifts=False
    ):
        self.T = T             
        self.max_T = max_T     
        self.Ndata = Ndata
        self.N = N                    
        self.ns = ns            
        self.rng = rng if rng is not None else np.random
        self.nfreq = nfreq      
        self.shift_range = shift_range      # control the frequency range of data
        self.max_shift = max_shift    
        self.shift_label = shift_label
        self.batchM_size = batchM_size
        self.shifts = shifts
        self.pow = pow
        
        if coef is None:
            coef = np.random.randn(Ndata,nfreq)
            nm = np.linalg.norm(coef, axis=1)
            coef = coef/(nm.repeat(nfreq).reshape(Ndata,nfreq))

        self.coef = coef
        
        freq_all=np.array([3,5,11,4,7,9,0,2,6,8,10,1])
        if nfreq > len(freq_all):
            print("n_freq must be less than ", len(freq_all))
            return
        freq = freq_all[0:nfreq]
        self.freqs = np.tile(freq,(Ndata,1))  
        
        #   In the batch, the same M(g) is used.  Set shuffle off in Dataloader. 
        if self.shift_label and (self.shifts is None):
            shifts=[]
            for i in range(Ndata):
                if i % batchM_size == 0:
                    shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)
                shifts.append(shift)
            self.shifts=np.array(shifts)        # shifts are given in radian [0, 2pi)
    
        self.shared_transition = shared_transition
        if self.shared_transition:
            self.init_shared_transition_parameters()


    def init_shared_transition_parameters(self):
        self.shared_shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)


    def __len__(self):
        return self.Ndata


    def __getitem__(self, i):       # i-th data and its shifts (T shifts)
        freq = self.freqs[i,:]
        dt = 1.0/self.N
        obs_t = np.arange(0, self.N*dt, dt) # observed time  [0,1]/N
        lat_t = np.power(obs_t, self.pow)      # observed time domain: lat_t = obs_t^pow
                
        if self.shift_label:
            shift = self.shifts[i]          
        elif self.shared_transition:
            shift = self.shared_shift 
        else:
            shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)
            
        fvals = []
        for t in range(self.T):     # shift * t ( action up to T times)
            lat_t = np.power(obs_t, self.pow) - t*shift/(2*math.pi)
            lat_t = lat_t + (lat_t<0)*(1.0)          
            fobs_t = np.matmul(np.sin(np.outer(2*np.pi*lat_t,freq)), self.coef[i,:])      # fn_t(j) = f_lat((t_j)**pow - shift*t)  
            fvals.append(fobs_t + self.ns * np.random.randn(self.N))

        if self.shift_label:
            return [fvals, shift]    # List (length T) of N-dim function values and shift 
        else:
            return fvals        # List (length T) of N-dim function values and time 
               


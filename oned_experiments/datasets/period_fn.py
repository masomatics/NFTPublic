# Data generation for neural fft
# Sequence of periodic functions on circle [0, 2\pi ) 
# shift is acted to a function 
# Modified from seq_mnist.py

import os
import numpy as np
import math
from copy import deepcopy
import pdb


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
            random_shifts = False,   # If true, each data has different shifts over epochs
            freq_fix = False,
            freq_manual = [],
            freqseed=1,
            test=0,  #if >0, this will be used for a random seed of the coefficients
            smallfreqs_num=0,
            smallfreq_strength=0
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
        self.smallfreqs_num = smallfreqs_num
        if smallfreqs_num > 0:
            self.smallfreq_strength = smallfreq_strength/smallfreqs_num
        else:
            self.smallfreq_strength = 0
        
#        self.freq = np.random.randint(0,N,nfreq)
        if coef is None:
            coef = np.random.randn(nfreq)
            coef = coef/np.linalg.norm(coef)
            #coef = coef/nfreq
        self.coef = coef
        self.coefs = []
        
        dt = 1.0/N
        t = np.arange(0, N*dt, dt) # time domain 
        
        if freq_fix == True:
            if len(freq_manual) >0:
                self.freqsel = freq_manual
            else:
                np.random.seed(freqseed)
                self.freqsel = np.random.randint(0,np.ceil(N/(5*self.shift_range)),nfreq) 
            print(self.freqsel)
        else:
            print('random freqs')


        # Generation of data (size: Ndata, dim: N)
        fdata =[]
        if test > 0:
            np.random.seed(test)


        for i in range(Ndata):
            # Initial function data generated: f(t) = \sum_j coef_j * sin( 2 \pi freqsel_j t)
            if freq_fix == True:
                freqsel = self.freqsel
                coef = np.random.randn(nfreq)
                coef = coef/np.linalg.norm(coef)
                if self.smallfreqs_num > 0:
                    coef[-self.smallfreqs_num:] = self.smallfreq_strength * coef[-self.smallfreqs_num:]

                self.coefs.append(coef)
            else:    
                freqsel = np.random.randint(0,np.ceil(N/(5*self.shift_range)),nfreq)  # randomly selected frequencies
                coef = self.coef
           
            
            f =  np.matmul(np.sin(np.outer(2*np.pi*t,freqsel)), coef)
            fdata.append(f)
            #fdata.append (f + ns * np.random.randn(N))
                       
        
        self.data = np.array(fdata)     # data:  Ndata x N (double array)      
        self.coefs = np.array(self.coefs)
        
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
            shift_label=True,
            batchM_size=1,
            shift_range=2,
            max_shift=[0.0, 2 * math.pi / 2],   # range of shift action (in radian)
            max_T=5,
            shared_transition=False,
            rng=None,
            nfreq=5,        # Number of selected frequecy to make a function
            coef=None,      # Coefficients of the selected frequencies. 
            ns=0.0,          # Noise level of additive Gaussian noise
            pow=3,
            shifts=None,
            freq_fix = False,
            freq_manual = [],
            freqseed = 1,
            test = 0,
            track_idx=False,
            smallfreqs_num=0,
            smallfreqs_strength=0
    ):
        self.T = T             
        self.max_T = max_T     
        self.Ndata = Ndata
        self.N = N                    
        self.ns = ns            
        self.rng = rng if rng is not None else np.random
        self.nfreq = nfreq + smallfreqs_num
        self.shift_range = shift_range      # control the frequency range of data
        self.max_shift = max_shift    
        self.shift_label = shift_label
        self.batchM_size = batchM_size
        self.shifts = shifts
        self.pow = pow
        self.freq_fix = freq_fix
        self.track_idx = track_idx
        self.smallfreqs_num = smallfreqs_num
        if smallfreqs_num > 0:
            self.smallfreqs_strength = smallfreqs_strength/smallfreqs_num
        else:
            self.smallfreqs_strength = 0

        if coef is None:
            coef = np.random.randn(self.nfreq)
            coef = coef/np.linalg.norm(coef)
            #coef = coef/nfreq
        self.coef = coef
        

        if freq_fix == True:

            if len(freq_manual) > 0:
                self.freqsel = freq_manual
            else:
                np.random.seed(freqseed)
                self.freqsel = np.random.randint(0,np.ceil(N/(5*self.shift_range)),self.nfreq) 
            print(self.freqsel)
        else:
            print('random freqs')

        #dt = 1.0/N
        #obs_t = np.arange(0, N*dt, dt) # observed time  [0,1]/N
        #lat_t = np.power(obs_t, self.pow)      # observed time domain: lat_t = obs_t^pow
        
        # In this class (nonlinear), unlike Shifted_freqFun(), data set is not contained in the class, because the time point must be calculated by shifts. 
        #   Instead, the frequencies and coefficients to make the latent functions are contained in the class
        # Generation of data (size: Ndata)
        #fdata =[]
        freqs = []
        coefs = [] 

        if test > 0:
            np.random.seed(test)
        print(test)

        for i in range(Ndata):

            if freq_fix == True:
                coef = np.random.randn(self.nfreq)
                coef = coef/np.linalg.norm(coef)
                if self.smallfreqs_num > 0:
                    coef[-self.smallfreqs_num:] = self.smallfreqs_strength * coef[-self.smallfreqs_num:]
                coefs.append(coef)
                freqs.append(self.freqsel)

            else:
                freqsel = np.random.randint(0,np.ceil(N/(5*self.shift_range)),self.nfreq)  # randomly selected frequencies
                freqs.append(freqsel)
                coefs.append(self.coef)
                      
   
        self.freqs = np.array(freqs)     # self.freqs:  Ndata x nfreq (double array) contains frequencoes to make the latent functions
        #  To make the function values use f =  np.matmul(np.sin(np.outer(2*np.pi*t,self.freqs[i,:])), self.coef)  # function value at latent t
        #self.lat_t = lat_t
        #self.obs_t = obs_t   
        self.coefs = np.array(coefs)   
        
        #   In the batch, the same M(g) is used.  Set shuffle off in Dataloader. 
        if self.shift_label and (self.shifts is None):
            shifts=[]
            for i in range(Ndata):
                if i % batchM_size == 0:
                    shift = self.rng.uniform(self.max_shift[0], self.max_shift[1],size=1)
                shifts.append(shift)
            self.shifts=np.array(shifts)        # shifts are given in radian [0, 2pi)
    
        # random shift 
        # self.shift = self.max_shift[0] + np.random.rand(1) * (self.max_shift[1] - self.max_shift[0])


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
            fobs_t = np.matmul(np.sin(np.outer(2*np.pi*lat_t,freq)), self.coefs[i])      # fn_t(j) = f_lat((t_j)**pow - shift*t)  
            fvals.append(fobs_t)

        if self.shift_label:
            if self.track_idx:
                shift = (shift, i) 
            return [fvals, shift]    # List (length T) of N-dim function values and shift 
        else:
            return fvals        # List (length T) of N-dim function values and time 
               


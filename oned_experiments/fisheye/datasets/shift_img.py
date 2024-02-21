import os
import numpy as np
import cv2
import torch
import torchvision
import math
import colorsys
from skimage.transform import resize
from copy import deepcopy
from utils.misc import get_RTmat
from utils.misc import freq_to_wave
import pdb
import copy
from defisheye import Defisheye
import torchvision.datasets as datasets
import torch.nn.functional as F


def shift(image, vshift, hshift):
    vlen = image.shape[0]
    hlen = image.shape[1]
    vrange = np.array(range(vlen))
    hrange = np.array(range(hlen))
    
    vrange = (vrange + vshift) % vlen
    hrange = (hrange + hshift) % vlen

    return image[:, tuple(hrange)][tuple(vrange), :]
    

def scale_and_deform(im, scale=20,  dtype = 'linear', format = 'fullframe', fov = 180, pfov = 140):
    #print(type(im))
    #im = torch.tensor(im)
    im = im.unsqueeze(0)
    upsampled_img = F.interpolate(im.permute([0, 3, 1,2]) , scale_factor=scale, mode='nearest')[0].permute([1,2,0])
    upsampled_img= Defisheye(np.array(upsampled_img),  dtype = 'linear', format = 'fullframe', fov = fov, pfov = pfov)
    upsampled_img= torch.tensor(upsampled_img.convert()).unsqueeze(0)
    img = F.interpolate(upsampled_img.permute([0, 3, 1,2]) , scale_factor=1./scale, mode='nearest')[0].permute([1,2,0])
    return img

# def deform_shift(image, shiftY, shiftX,):
    
#     obj = Defisheye(shift(image, shiftY, shiftX), dtype=dtype, format=format, fov=fov, pfov=pfov)
#     shift_new_image = obj.convert()
#     return shift_new_image

def deform_batch(images, scale=20,  dtype = 'linear', format = 'fullframe', fov = 180, pfov = 140):
    imgs=[]
    upsampled_img = F.interpolate(images.permute([0,3, 1,2]) , scale_factor=20, mode='nearest').permute([0, 2,3, 1])
    for t in range(len(upsampled_img)):
        obj = Defisheye(np.array(upsampled_img[t]),  dtype = 'linear', format = 'fullframe', fov = fov, pfov = pfov)
        deformed_img = torch.tensor(obj.convert())
        imgs.append(deformed_img)
    deformed_imgs = torch.stack(imgs) 
    deformed_imgs = F.interpolate(deformed_imgs.permute([0,3, 1,2]) , scale_factor=1./20, mode='nearest').permute([0, 2,3, 1])
    return deformed_imgs


class Shift_cifar():
    # Rotate around z axis only.

    def __init__(
            self,
            root='../data/',
            train=True,
            transforms=torchvision.transforms.ToTensor(),
            T=3,
            max_vshift=[-10, 10],
            max_hshift=[-10, 10],
            label=False,
            label_velo=False,
            label_accl=False,
            max_T=9,
            shared_transition=False,
            deform=False,
            fov = 180,
            pfov = 140,
            scale_at_deform=1,
            rng=None, **kwargs
    ):
        self.T = T
        self.max_T = max_T
        self.rng =  rng if rng is not None else np.random
        self.transforms = transforms
        self.h_velocity_range = (-max_hshift, max_hshift) if isinstance(max_hshift, (int, float)) else max_hshift  
        self.v_velocity_range = (-max_vshift, max_vshift) if isinstance(max_vshift, (int, float)) else max_vshift  
        self.deform = deform
        self.scale_at_deform = scale_at_deform

        self.label = label
        self.label_velo = label_velo

        alldat = datasets.CIFAR100(root=root, train=train, download=True)
        self.data = alldat.data
        self.labels = alldat.targets
        self.fov = fov
        self.pfov = pfov

        self.shared_transition = shared_transition
        if self.shared_transition:
            self.init_shared_transition_parameters()

    def init_shared_transition_parameters(self):

        self.shift_h = self.rng.uniform(self.h_velocity_range)
        self.shift_v = self.rng.uniform(self.v_velocity_range)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        image = np.array(self.data[i], np.float32)/255.

        if self.shared_transition:
            (h_velo, v_velo) = (self.shift_h, self,shift_v)
        
        else:
            h_velo = self.rng.uniform(self.h_velocity_range[0], self.h_velocity_range[1]) 
            v_velo = self.rng.uniform(self.v_velocity_range[0], self.v_velocity_range[1]) 

        images = []
        for t in range(self.T):
            h_shift = np.floor(h_velo*t).astype(int)
            v_shift = np.floor(v_velo*t).astype(int) 
            #print(h_shift, v_shift)
            shifted_image = torch.tensor(shift(image, v_shift, h_shift))

            if self.deform:
                shifted_image = scale_and_deform(shifted_image, 
                scale=self.scale_at_deform,  dtype = 'linear', format = 'fullframe', fov = self.fov, pfov =self.pfov)

            shifted_image = shifted_image.permute([2,0,1])
            images.append(shifted_image)
        

        if self.label or self.label_velo:
            ret = [images]
            if self.label:
                ret += [self.targets[i]]
            if self.label_velo:
                ret += [[h_velo, v_velo]]
            return ret
        else:
            return images
            
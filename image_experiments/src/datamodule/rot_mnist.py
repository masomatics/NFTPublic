import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
import math
import random


class Rot_MNIST(torch.utils.data.Dataset):
    occlusion_height = 14
    occlusion_width = 14

    def __init__(self, split, occlusion, num_views, point_symmetry, root, gridsample=False, stationary=False):
        if split == 'train':
            self.dataset = MNIST(root=root, download=True, train=True)
        elif split == 'test':
            self.dataset = MNIST(root=root, download=True, train=False)
        elif split == 'ood':
            self.dataset = FashionMNIST(root=root, download=True, train=False)
        elif split == 'ood_train':
            self.dataset = FashionMNIST(root=root, download=True, train=True)
        elif split == 'ood_test':
            self.dataset = FashionMNIST(root=root, download=True, train=False)
        elif split == 'ood2':
            self.dataset = KMNIST(root=root, download=True, train=False)
        elif split == 'ood2_train':
            self.dataset = KMNIST(root=root, download=True, train=True)
        elif split == 'ood2_test':
            self.dataset = KMNIST(root=root, download=True, train=False)
        self.occlusion = occlusion
        self.num_views = num_views
        self.point_symmetry = point_symmetry
        self.gridsample = gridsample
        self.stationary = stationary

    def apply_occlusion(self, img):
        img[..., :self.occlusion_height, :] = 0
        img[..., :, :self.occlusion_width] = 0
        return img

    def random_rotate(self, img, rad=None):
        angle = math.degrees(rad) if rad is not None else random.random() * 360
        img = TF.rotate(img, angle)
        rad = math.radians(angle)
        return img, rad

    def transform(self, img, param):
        img, param = self.random_rotate(img, param)
        return img, param

    def sample_param(self, view_id, base_rad=None):
        if self.gridsample:
            return 2 * math.pi * view_id / self.num_views
        if self.stationary:
            return (view_id + 1) * base_rad
        return 2 * math.pi * random.random()

    def __getitem__(self, index):
        img, target = self.dataset.__getitem__(index)
        img = TF.to_tensor(img)        
        if self.point_symmetry and target >= 5:
            halfsize = img.shape[-1] // 2
            img[..., :, halfsize:] = img[..., :, :halfsize].flip(dims=(-1,-2))
        imgs = []
        params = []
        base_rad = random.uniform(-0.5 * math.pi, 0.5 * math.pi) if self.stationary else None
        for i in range(self.num_views + 1):
            param = self.sample_param(i, base_rad)
            _img, _param = self.transform(img, param)
            if self.occlusion and i != self.num_views:
                _img = self.apply_occlusion(_img)
            imgs.append(_img)
            params.append(_param)
        
        return imgs, params, target

    def __len__(self):
        return self.dataset.__len__()
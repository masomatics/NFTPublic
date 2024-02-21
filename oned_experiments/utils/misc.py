import math
import torch
from torch import nn
from einops import repeat
import numpy as np


def freq_to_wave(freq, is_radian=True):
    _freq_rad = 2 * math.pi * freq if not is_radian else freq
    return torch.hstack([torch.cos(_freq_rad), torch.sin(_freq_rad)])


def unsqueeze_at_the_end(x, n):
    return x[(...,) + (None,)*n]


def get_RTmat(theta, phi, gamma, w, h, dx, dy):
    d = np.sqrt(h ** 2 + w ** 2)
    f = d / (2 * np.sin(gamma) if np.sin(gamma) != 0 else 1)
    # Projection 2D -> 3D matrix
    A1 = np.array([[1, 0, -w / 2],
                   [0, 1, -h / 2],
                   [0, 0, 1],
                   [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

    RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                   [0, 1, 0, 0],
                   [np.sin(phi), 0, np.cos(phi), 0],
                   [0, 0, 0, 1]])

    RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                   [np.sin(gamma), np.cos(gamma), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, f],
                  [0, 0, 0, 1]])
    # Projection 3D -> 2D matrix
    A2 = np.array([[f, 0, w / 2, 0],
                   [0, f, h / 2, 0],
                   [0, 0, 1, 0]])
    return np.dot(A2, np.dot(T, np.dot(R, A1)))


def create_reportdict(loss , loss_dict):
    report_dict = {'train/loss' : torch.tensor(loss).item}

    for key in list(loss_dict.keys()):
        lossname = key.split('_')[-1]
        keyname = f"""train/loss_{lossname}"""
        report_dict[keyname] = torch.tensor(loss_dict[key]).item()
    return report_dict


def specnorm(weightmat, repeat=5):
    mydim0 = weightmat.shape[0]
    mydim1 = weightmat.shape[1]

    device = weightmat.device
    random_vec_r = torch.tensor(np.random.uniform(size=(mydim1, 1))).float().to(device)
    random_vec_l = torch.tensor(np.random.uniform(size=(mydim0, 1))).float().to(device)

    for k in range(repeat):
        random_vec_l = weightmat @ random_vec_r
        random_vec_l = random_vec_l / torch.sqrt(torch.sum(random_vec_l ** 2))

        random_vec_r = weightmat.permute([1, 0]) @ random_vec_l
        random_vec_r = random_vec_r / torch.sqrt(torch.sum(random_vec_r ** 2))

    val = random_vec_l.permute([1, 0]) @ weightmat @ random_vec_r
    return val


def scale_specnorm(linlayer, const):
    snorm = specnorm(nn.Parameter(linlayer.weight.data))
    linlayer.weight.data = nn.Parameter(const * linlayer.weight.data / snorm)
    return linlayer

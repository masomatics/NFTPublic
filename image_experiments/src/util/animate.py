'''
Usage: python src/util/animate.py ${log_dir} ${output_dir}
Example: python src/util/animate.py log/co3d/single_seq/nbasis4_nview1_lowrank/torch.optim.AdamW_bs64_lr0.0001_wd0.05/enc4-9-512-12/dec4-3-256-6/version_0/ figure/co3d/single_seq
'''
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import os
import glob
from pathlib import Path
import subprocess
import math
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.utils import make_grid
from omegaconf import DictConfig
from hydra.utils import instantiate
# from src.train import LitEquivAE, instantiate_model


def put_src(obj, key='_target_'):
    for k, v in obj.items():
        if isinstance(v, DictConfig):
            obj[k] = put_src(v, key)
    if key in obj:
        if ('model' in obj[key] or 'datamodule' in obj[key]) and not 'src' in obj[key]:
            obj[key] = f'src.{obj[key]}'
    return obj


def load_config_and_model(log_path):
    ckpt_file = glob.glob(os.path.join(log_path, 'checkpoints/*.ckpt'))[0]
    checkpoint = torch.load(ckpt_file)
    config = put_src(checkpoint['hyper_parameters'])
    config = DictConfig(config)
    # model = instantiate_model(config)
    model = instantiate(config.model)
    model.load_state_dict(checkpoint['state_dict'])

    # disable randomness, dropout, etc...
    model.eval()
    return config, model


def seq_pattern_1(seq_length):
    quats = []
    seq_length = seq_length // 3
    for j in range(3): # for x, y, z axes
        for i in range(seq_length):
            rad = 2 * torch.pi * i / seq_length
            quat = torch.zeros(1, 4)
            quat[:, 0] = math.cos(rad / 2)
            quat[:, j + 1] = math.sin(rad / 2)
            quats.append(quat)
    return quats

# def seq_pattern_2(seq_length):
#     quats = []
#     for i in range(seq_length):
#         theta = (i / seq_length) * 0.5 * torch.pi
#         gamma = (i / seq_length) * 4 * torch.pi 
#         quat = torch.zeros(1, 4)
#         # quat[:, 1] = math.sin(theta) * math.sin(gamma)
#         # quat[:, 2] = math.sin(theta) * math.cos(gamma)
#         quat[:, 1] = math.cos(theta)
#         quat[:, 2] = math.sin(theta)
#         quat[:, 3] = math.sin(theta)
#         quat /= quat.norm(dim=1)
#         quats.append(quat)
#     return quats

def seq_pattern_2(seq_length):
    quats = []
    for i in range(seq_length):
        theta = (i / seq_length) * torch.pi 
        gamma = (i / seq_length) * 0.2 * torch.pi 
        quat = torch.zeros(1, 4)
        quat[:, 0] = math.cos(theta)
        quat[:, 1] = math.sin(theta) * math.sin(gamma)
        quat[:, 2] = math.sin(theta) * math.sin(gamma)
        quat[:, 3] = math.sin(theta) * math.cos(gamma)
        quat /= quat.norm(dim=1)
        quats.append(quat)
    return quats    

def seq_pattern_3(seq_length, elevation=None):
    quats = []
    for i in range(seq_length):
        rad = i * 2 * torch.pi / seq_length
        if elevation is None:
            elevation = math.radians(15)
        quat = torch.zeros(1, 4)
        quat[:, 0] = math.cos(elevation)
        quat[:, 1] = -math.sin(elevation) * math.cos(rad)
        quat[:, 2] = math.sin(elevation) * math.sin(rad)
        quats.append(quat)
    return quats

def quat_seq(seq_length, pattern):
    if pattern == 1:
        return seq_pattern_1(seq_length)
    else:
        return seq_pattern_2(seq_length)

def plot_sequence_for_gif(dataset, _model, seq_length=30, batch_size=8, out_dir='../figure', fig_name='animation', pattern=1):
    model = _model.cpu().eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    xs, params, _ = next(iter(dataloader))
    # xs = list(xs.unbind(dim=1))
    # params = list(params.unbind(dim=1))
    
    x, param = xs.pop(), params.pop()
    img_size = x.shape[-1]
    z = model.latent_pooling(xs, params)
    
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    path_png = Path(path / 'png')
    path_png.mkdir(parents=True, exist_ok=True)
    
    for figure_index, quat in enumerate(quat_seq(seq_length, pattern)):
        quat_next = quat * torch.ones_like(param)
        gen_xs = []
        for pickup_degree in range(len(model.actions) + 1):
            _z = z.clone()
            if pickup_degree < len(model.actions):
                for degree, z_d in enumerate(model.to_chunks(_z)):
                    if degree != pickup_degree:
                        z_d[:] = 0
            x_next = model.dec(model.trans(_z, [quat_next]))
            gen_xs.append(x_next)
        x_concat = torch.stack(xs + gen_xs)
        img_grid = make_grid(x_concat.reshape(-1, 3, img_size, img_size), nrow=batch_size, normalize=True)
        plt.axis('off')
        plt.imshow(np.transpose(img_grid, (2, 1, 0)))
        plt.savefig(path_png / f'{figure_index:04d}.png', dpi=120)
    
    subprocess.run(f'convert -delay 10 -loop 0 {path_png}/*.png {path}/{fig_name}.gif'.split(' '))


# python src/util/animate.py --log-dir log/abo/vit_b/nbasis4_nview1_lowrank/torch.optim.AdamW_bs48_lr0.0001_wd0.05/enc4-12-768-12/dec4-3-256-6/version_5/ --out-dir figure/abo/vit_b/ --batch-size 4 --split test --pattern 2
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--split", type=str, default='')
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--pattern", type=int, default=1)
    args = parser.parse_args()

    config, model = load_config_and_model(args.log_dir)
    if args.split:
        dataset = instantiate(config.data.dataset, split=args.split)
    else:
        dataset = instantiate(config.data.dataset, train=False)
    plot_sequence_for_gif(dataset, model, batch_size=args.batch_size, seq_length=args.seq_len, out_dir=args.out_dir, fig_name=config.tag, pattern=args.pattern)
import math
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

def show(img):
    img = img.detach()
    img = F.to_pil_image(img)
    return plt.imshow(np.asarray(img), animated=True)

def animate_gridsample(dataset, _model, gif_dir, gif_name, seq_length=30, batch_size=8):
    model = _model.cpu().eval()
    dataset.gridsample = True
    dataset.num_views = seq_length
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    xs, params, _ = next(iter(dataloader))
    
    x_target, param_target = xs.pop(), params.pop()
    
    fig = plt.figure()
    # fig.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ims = []
    for x, param in zip(xs, params):
        gen_xs = []
        z = model.latent_pooling([x], [param])
        for pickup_degree in range(len(model.actions) + 1):
            _z = z.clone()
            if pickup_degree < len(model.actions):
                for degree, z_d in enumerate(model.to_chunks(_z)):
                    if degree != pickup_degree:
                        z_d[:] = 0
            x_next = model.dec(model.trans(_z, [param_target]))
            gen_xs.append(x_next)
        gen_xs = [x] + gen_xs + [x_target]
        x_concat = torch.stack(gen_xs, dim=1).flatten(0, 1)
        img_grid = torchvision.utils.make_grid(x_concat, nrow=len(gen_xs), normalize=True)
        frame = show(img_grid)
        ims.append([frame])
    
    path = Path(gif_dir)
    path.mkdir(parents=True, exist_ok=True)
    path_gif = Path(path / gif_name)
    plt.axis('off')
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save(path_gif, writer="imagemagick", dpi=100)
    #ani.save('anim.mp4', writer="Pillow")
    plt.show()
    return ani

def get_denormalize(config):
    data_mean = config.data.dataset.transform.transforms[-1].mean
    data_std = config.data.dataset.transform.transforms[-1].std
    denormalize = T.Compose([ 
        T.Normalize(mean=[0, 0, 0],
                    std=[1 / std for std in data_std]),
        T.Normalize(mean = [ -mean for mean in data_mean ],
                    std = [ 1., 1., 1. ]),
    ])
    return denormalize

def animate_movez(dataset, config, _model, gif_dir, gif_name, seq_length=30, batch_size=8):
    model = _model.cpu().eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    xs, params, _ = next(iter(dataloader))
    denormalize = get_denormalize(config)
    
    x, param = xs[0], params[0]
    # z = model.latent_pooling([x], [param])
    z = model.enc(x)
    
    fig = plt.figure()
    ims = []
    quats = poseseq_rotsideview(seq_length, param)
    for quat in quats:
        gen_xs = [x]
        for pickup_degree in range(len(model.actions) + 1):
            _z = model.trans(z, quat)
            if pickup_degree < len(model.actions):
                for degree, z_d in enumerate(model.to_chunks(_z)):
                    if degree != pickup_degree:
                        z_d[:] = 0
            x_next = model.dec(_z)
            gen_xs.append(x_next)
        # x_concat = torch.cat(gen_xs, dim=0)
        x_concat = torch.stack(gen_xs, dim=1).flatten(0, 1)
        x_concat = denormalize(x_concat)
        img_grid = torchvision.utils.make_grid(x_concat, nrow=len(gen_xs), normalize=False)
        frame = show(img_grid)
        ims.append([frame])
    
    path = Path(gif_dir)
    path.mkdir(parents=True, exist_ok=True)
    path_gif = Path(path / gif_name)
    plt.axis('off')
    plt.rcParams["savefig.bbox"] = 'tight'
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save(path_gif, writer="imagemagick", dpi=200)
    #ani.save('anim.mp4', writer="Pillow")
    plt.show()
    return ani

def animate_gridz(dataset, config, models, gif_dir, gif_name, seq_length=24, batch_size=8):
    models = [model.cpu().eval() for model in models]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    xs, params, _ = next(iter(dataloader))
    denormalize = get_denormalize(config)
    
    x, param = xs[0], params[0]
    # zs = [model.latent_pooling([x], [param]) for model in models]
    zs = [model.enc(x) for model in models]

    # quat = torch.zeros_like(param)
    # quat[:, 0] = math.cos(0.5 * torch.pi / 2)
    # quat[:, 1] = math.sin(0.5 * torch.pi / 2)
    # zs = [model.trans(z, quat) for model, z in zip(models, zs)]
    
    fig = plt.figure()
    ims = []
    quats = poseseq_rotsideview(seq_length, param)
    for quat in quats:
        # rad = i * 3 * 2 * torch.pi / seq_length
        # quat = torch.zeros(1, 4)
        # quat[:, 0] = math.cos(rad / 2)
        # quat[:, (i // (seq_length // 3)) + 1] = math.sin(rad / 2)

        # rad = i * 2 * torch.pi / seq_length
        # elevation = math.radians(15)
        # quat = torch.zeros_like(param)
        # quat[:, 0] = math.cos(elevation)
        # quat[:, 1] = -math.sin(elevation) * math.cos(rad)
        # quat[:, 2] = math.sin(elevation) * math.sin(rad)


        _param = quat
        gen_xs = []
        for model, z in zip(models, zs):
            x_pred = model.dec(model.trans(z, _param))
            gen_xs.append(x_pred)
        gen_xs = [x] + gen_xs
        x_concat = torch.cat(gen_xs, dim=0)
        x_concat = denormalize(x_concat)
        x_concat = torch.clamp(x_concat, 0, 1)
        img_grid = torchvision.utils.make_grid(x_concat, nrow=batch_size)
        frame = show(img_grid)
        ims.append([frame])
    
    path = Path(gif_dir)
    path.mkdir(parents=True, exist_ok=True)
    path_gif = Path(path / gif_name)
    plt.axis('off')
    plt.rcParams["savefig.bbox"] = 'tight'
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save(path_gif, writer="imagemagick", dpi=200)
    #ani.save('anim.mp4', writer="Pillow")
    plt.show()
    return ani

def poseseq_rotsideview(seq_length, param, zoom=True):
    quats = []
    for i in range(seq_length):
        rad = i * 2 * torch.pi / seq_length
        quat = torch.zeros_like(param)
        quat[:, 0] = math.cos(rad / 2)
        quat[:, 3] = math.sin(rad / 2)
        if zoom:
            quat[:, -1] = 1 + 0.5 * math.sin(rad)
        quats.append(quat)
    return quats
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from einops import repeat
from pathlib import Path
from tqdm import tqdm

def get_imgs(dataset, inds):
    imgs = dataset.imgs[inds].div(255)
    imgs = dataset.transform(imgs)
    return imgs
    
def pred_sequence(model, dataset, input_ind, target_inds=None, trans_params=None, freq=-1,
                  canonical=True):
    # img = get_imgs(dataset, input_ind).unsqueeze(dim=0)  # 1 C H W
    # if hasattr(dataset, 'quats'):
    #     param = dataset.quats[input_ind].unsqueeze(dim=0)
    # elif hasattr(dataset, 'quats'):

    # target_imgs = get_imgs(dataset, target_inds)
    # target_quats = dataset.quats[target_inds]
    (img, _), (param, _), _ = dataset[input_ind]
    img = img.unsqueeze(dim=0)
    if not hasattr(param, '__len__'):
        param = torch.Tensor([param])
    else:
        param = param.unsqueeze(dim=0)
    
    z = model.enc(img)  # 1 L A
    if canonical:
        inv_param = model.invert_param(param)
        z = model.trans(z, inv_param)
    
    # mask latent for non-target freq
    for l, z_l in enumerate(model.to_chunks(z)):
        if freq < 0:
            break
        if l != freq:
            z_l[:] = 0
    
    if target_inds is not None:
        target_imgs = []
        target_params = []
        for target_ind in target_inds:
            (target_img, _), (target_param, _), _ = dataset[target_ind]
            target_imgs.append(target_img)
            target_params.append(target_param)
        target_imgs = torch.stack(target_imgs)
        
        if not hasattr(target_param, '__len__'):
            target_params = torch.Tensor(target_params)
        else:
            target_params = torch.stack(target_params)
    
        z = repeat(z, 'b h a -> (view b) h a', view=len(target_inds))
    else:
        target_imgs = []
        target_params = trans_params
    z = model.trans(z, target_params)
    pred_imgs = model.dec(z)
    return img, pred_imgs, target_imgs

def plot_pred_sequence(model, dataset, input_ind, target_inds, target_freqs, 
                       x_offset=15, y_offset=15, fig_path=None, ylab=True):
    pred_imgs = []
    # for freq in range(-1, 5):
    for freq in target_freqs:
        input_img, preds, target_imgs = pred_sequence(model, dataset, input_ind, target_inds, freq=freq)
        preds = torch.cat([torch.zeros_like(input_img), preds])
        pred_imgs.append(preds)

    imgs = pred_imgs + [input_img, target_imgs]
    imgs = torch.cat(imgs, dim=0)
    
    img_size = imgs.shape[-1] + 2 # 2 is the padding by make_grid
    nrow = len(target_inds) + 1
    grid_imgs = make_grid(imgs, normalize=True, nrow=nrow)
    grid_imgs[..., :(len(target_freqs) * img_size), :img_size] = 0
    plt.imshow(grid_imgs.permute(1, 2, 0))
    plt.axis('off')

    hsize = img_size // 2
    plt.text(hsize, (len(target_freqs) + 1) * img_size + y_offset, 'input', ha='center')
    if ylab:
        if len(target_freqs) == 1:
            plt.text(img_size * nrow + x_offset, hsize, 'pred', rotation=-90, va='center')
            plt.text(img_size * nrow + x_offset, hsize + img_size, 'GT', rotation=-90, va='center')
        else:
            plt.text(img_size * nrow + x_offset, len(target_freqs) * img_size // 2, 'prediction\n', rotation=-90, va='center')
            for i, freq in enumerate(target_freqs):
                if i == 0:
                    text = f'freq={freq}'
                elif freq == -1:
                    text = 'all'
                else:
                    text = f'{freq}'
                plt.text(img_size * nrow + x_offset, hsize + img_size * i, text, 
                        rotation=-90, va='center')
            plt.text(img_size * nrow + x_offset, hsize + len(target_freqs) * img_size, 'GT\n', rotation=-90, va='center')
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', dpi=400)
    
    
def video_pred_sequence(model, dataset, input_ind, target_freqs, target_params, 
                        x_offset=15, y_offset=15, fig_dir=None, lab=True, fontsize=None):
    if fontsize is not None:
        plt.rcParams["font.size"] = fontsize
    
    for p, target_param in enumerate(tqdm(target_params)):
        imgs = []
        for freq in target_freqs:
            input_img, preds, _ = pred_sequence(model, dataset, input_ind, target_inds=None, 
                                                trans_params=target_param, freq=freq, canonical=False)
            imgs.append(preds)
        imgs = [input_img] + imgs
        imgs = torch.cat(imgs, dim=0)
        
        img_size = imgs.shape[-1] + 2 # 2 is the padding by make_grid
        nrow = len(target_freqs) + 1
        grid_imgs = make_grid(imgs, normalize=True, nrow=nrow)
        plt.imshow(grid_imgs.permute(1, 2, 0))
        plt.axis('off')

        if lab:
            hsize = img_size // 2
            plt.text(hsize, -y_offset, 'input\n', ha='center')
            # plt.text(len(target_freqs) * img_size // 2, -y_offset, 'prediction\n', ha='center')
            for i, freq in enumerate(target_freqs):
                if i == 0:
                    text = f'prediction\nfreq={freq}'
                elif freq == -1:
                    text = 'all'
                else:
                    text = f'{freq}'
                plt.text(hsize + img_size * (i + 1), -y_offset, text, ha='center')
        if fig_dir is not None:
            plt.savefig(Path(fig_dir) / f'{p:04}', bbox_inches='tight', dpi=200)
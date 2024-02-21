import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import os
import glob
import shutil

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
# from src.util.eval import linear_probe


def get_last_ckpt(logger):
    pattern = f'{logger.root_dir}/*/checkpoints/*.ckpt'
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    if len(ckpts) == 0:
        return None
    
    last_ckpt = ckpts[-1]
    return last_ckpt


def copy_cache(root, cache_dir):
    for file in glob.glob(os.path.join(root, '*.cache')):
        # cache_dir = os.path.join(cache_dir, '')
        os.makedirs(cache_dir, exist_ok=True)
        print(f'copy cache file {file} to {cache_dir}')
        shutil.copy(file, cache_dir)


def get_loaders(config: DictConfig):
    if config.test:
        train_loader = instantiate(config.data, dataset={'split': 'test'})
    else:
        train_loader = instantiate(config.data, dataset={'split': 'train'}, shuffle=True)
    test_loader = instantiate(config.data, dataset={'split': 'test'}, shuffle=True)
    return train_loader, test_loader

@hydra.main(version_base=None, config_path='../config', config_name='train')
def main(config: DictConfig):
    if config.seed is not None:
        seed_everything(config.seed, workers=True)
    model = instantiate(config.model)
    model.save_hyperparameters(config)
    trainer = instantiate(config.trainer)
    last_ckpt = get_last_ckpt(trainer.logger) if config.resume else None
    
    if trainer.global_rank == 0:
        print(OmegaConf.to_yaml(config))
        if hasattr(config.data.dataset, 'cache_dir')\
            and hasattr(config, 'copy_cache'):
            if config.copy_cache:
                copy_cache(config.data.dataset.root, config.data.dataset.cache_dir)

    train_loader, test_loader = get_loaders(config)
    val_dataloaders = [test_loader]
    if not hasattr(config, 'no_ood'):
        ood_loader = instantiate(config.data, dataset={'split': 'ood'}, shuffle=True)
        val_dataloaders.append(ood_loader)
    if hasattr(config, 'ood2'):
        ood_loader = instantiate(config.data, dataset={'split': 'ood2'}, shuffle=True)
        val_dataloaders.append(ood_loader)

    fit_kwargs = dict(model=model, 
                      train_dataloaders=train_loader, 
                      val_dataloaders=val_dataloaders)
    if last_ckpt is not None:
        fit_kwargs['ckpt_path'] = last_ckpt
        print(f'set to load model: {last_ckpt}')
    trainer.fit(**fit_kwargs)
    return model

if __name__ == '__main__':
    main()
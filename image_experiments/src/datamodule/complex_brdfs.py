import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import os
import io
import random
import glob
import shutil
from collections import defaultdict
import PIL
import pfio
from pfio.cache import MultiprocessFileCache

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.datamodule.abo import save_json, load_json

SPLITS = ['train', 'val', 'test', 'ood']
SPLIT_PROBS = [0.8, 0.1, 0.1, 0.05]


def create_meta_files(root):
    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    cls_to_id = defaultdict(lambda: len(cls_to_id))
    meta = {split: [] for split in SPLITS}
    same_orbit_indexes = {split: defaultdict(list) for split in SPLITS}
    index = {split: 0 for split in SPLITS}

    for cls in classes:
        cls_id = int(cls)
        ood_flag = random.random() <= SPLIT_PROBS[-1]
        paths = sorted(glob.glob(os.path.join(root, cls, 'rgba_*.png')))
        for path in paths:
            if ood_flag:
                split = 'ood' 
            else: 
                split = random.choices(SPLITS[:-1], weights=SPLIT_PROBS[:-1])
                split = split[0]
            
            view_id = int(path.split('/')[-1].split('.')[0].split('_')[-1])
            angle = (view_id / 24) * 2 * torch.pi

            meta[split].append([path, angle, cls_id])
            same_orbit_indexes[split][cls_id].append(index[split])
            index[split] += 1
    
    save_json(meta, os.path.join(root, 'meta.json'))
    save_json(same_orbit_indexes, os.path.join(root, 'same_orbit_indexes.json'))


class Complex_BRDFs(Dataset):
    def __init__(self, root, cache_dir='/tmp/cache', split=None, train=True, transform=None, num_views=1, gridsample=False):
        self.root = root
        self.cache_dir = cache_dir
        self.transform = transform
        self.num_views = num_views
        self.split = split
        self.cache_file_name = f'{self.split}.cache'
        
        if split is None:
            split = 'train' if train else 'test'

        meta_path = os.path.join(root, 'meta.json')
        if not os.path.exists(meta_path):
            create_meta_files(root)
        
        self.meta = load_json(meta_path)[split]

        ind_file = os.path.join(root, 'same_orbit_indexes.json')
        self.same_orbit_indexes = load_json(ind_file)[split]
        self.gridsample = gridsample

        self._cache = \
                MultiprocessFileCache(len(self), dir=cache_dir, do_pickle=True)
        if os.path.exists(self.cache_path()):
            self.load_cache()

    def cache_path(self):
        return os.path.join(self.cache_dir, self.cache_file_name)
    
    def local_cache_path(self):
        return os.path.join(self.root, self.cache_file_name)

    def _load_image(self, index):
        img_path = self.meta[index][0]
        img_path = os.path.join(self.root, img_path)
        with pfio.v2.open_url(img_path, 'rb') as f:
            x = bytearray(f.read())
            x = PIL.Image.open(io.BytesIO(x)).convert('RGB')
            # x = TF.resize(x, self.img_size)
        return x

    def _get_instance(self, index):
        img = self._cache.get_and_cache(index, self._load_image)
        # img = self._load_image(index)
        if self.transform is not None:
            img = self.transform(img)
        _, angle, cls = self.meta[index]
        return img, angle, cls

    def __getitem__(self, index):
        img, angle, cls = self._get_instance(index)
        same_orbit_inds = self.same_orbit_indexes[str(cls)]
        if self.gridsample:
            sampled_inds = same_orbit_inds
        else:
            sampled_inds = random.choices(same_orbit_inds, k=self.num_views)
        imgs = [img]
        angles = [angle]
        for ind in sampled_inds:
            img, angle, _ = self._get_instance(ind)
            imgs.append(img)
            angles.append(angle)
        
        return imgs, angles, cls

    def __len__(self):
        return len(self.meta)

    def save_cache(self):
        self._cache.preserve(self.cache_file_name)
        shutil.copy(self.cache_path(),
                    self.local_cache_path())

    def copy_cache(self):
        return shutil.copy(self.local_cache_path(),
                           self.cache_path())

    def load_cache(self):
        self._cache.preload(self.cache_file_name)

if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from src.util.data_mean_and_std import data_stats

    root = sys.argv[1]
    trans = T.Compose([
        T.ToTensor(),
    ])
    for split in ['train', 'test', 'ood']:
        dataset = Complex_BRDFs(root, split=split, transform=trans)
        loader = DataLoader(dataset, batch_size=512, num_workers=8)
        stats = data_stats(loader)
        print(f'mean: {[v / 255 for v in stats.mean]}')
        print(f'std: {[v / 255 for v in stats.stddev]}')
        dataset.save_cache()
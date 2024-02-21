import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import os
import io
import glob
import random
import shutil
from collections import defaultdict
import json
import PIL
import pfio
from pfio.cache import MultiprocessFileCache

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.datasets.folder import find_classes
from src.util.quaternion import matrix_to_quaternion

SPLITS = ['train', 'val', 'test', 'ood']
SPLIT_PROBS = [0.8, 0.1, 0.1, 0.05]


def save_json(obj, path):
    with pfio.v2.open_url(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with pfio.v2.open_url(path, 'r') as f:
        return json.load(f)


def pose_to_quat(poses):
    quats = dict()
    for pose in poses:
        view = pose['index']
        Rt = torch.Tensor(pose['pose']).reshape(4, 4)
        rot = Rt[:3, :3]
        # trans = Rt[:3, 3].reshape(-1)
        quat = matrix_to_quaternion(rot)
        quats[view] = quat.numpy().tolist()
    return quats


def create_meta_files(root):
    classes, cls_to_id = find_classes(root)

    cat_to_id = defaultdict(lambda: len(cat_to_id))
    env_to_id = defaultdict(lambda: len(env_to_id))

    meta = {split: [] for split in SPLITS}
    same_orbit_indexes = {split: defaultdict(list) for split in SPLITS}
    same_orbit_env_indexes = {split: defaultdict(list) for split in SPLITS}
    index = {split: 0 for split in SPLITS}

    for cls in classes:
        metafile = os.path.join(root, cls, 'metadata.json')
        _meta = load_json(metafile)
        poses = _meta['views']
        quats = pose_to_quat(poses)
        env_names = _meta['envs']

        # cls_id = cls_to_id[cls]
        ood_flag = random.random() <= SPLIT_PROBS[-1]
        paths = sorted(glob.glob(os.path.join(root, cls, 'render/*/*.jpg')))
        for path in paths:
            if ood_flag:
                split = 'ood' 
            else: 
                split = random.choices(SPLITS[:-1], weights=SPLIT_PROBS[:-1])
                split = split[0]
            
            env = int(path.split('/')[-2])
            env_name = env_names[env]
            env_id = env_to_id[env_name]
            
            view_id, _ = os.path.splitext(path.split('_')[-1])
            view_id = int(view_id)

            quat = quats[view_id]

            # category = get_category(cls)
            meta[split].append([path, quat, cls, view_id, env_id])
            same_orbit_indexes[split][cls].append(index[split])
            same_orbit_env_indexes[split][f'{cls}.{env_id}'].append(index[split])
            index[split] += 1
    
    save_json(meta, os.path.join(root, 'meta.json'))
    save_json(same_orbit_indexes, os.path.join(root, 'same_orbit_indexes.json'))
    save_json(same_orbit_env_indexes, os.path.join(root, 'same_orbit_env_indexes.json'))
    save_json(cls_to_id, os.path.join(root, 'cls_to_id.json'))


class ABO_Material(Dataset):
    def __init__(self, root, cache_dir='/tmp/cache', split=None, train=True, transform=None, num_views=1, 
                 img_size=224, care_env=True, gridsample=False):
        self.root = root
        self.cache_dir = cache_dir
        self.transform = transform
        self.num_views = num_views
        self.img_size = img_size
        self.gridsample = gridsample
        self.split = split
        self.cache_file_name = f'{self.split}.cache'
        
        if split is None:
            split = 'train' if train else 'test'

        meta_path = os.path.join(root, 'meta.json')
        if not os.path.exists(meta_path):
            create_meta_files(root)
        
        self.meta = load_json(meta_path)[split]

        if care_env:
            ind_file = os.path.join(root, 'same_orbit_env_indexes.json')
            self.get_orbit_key = lambda cls, env: f'{cls}.{env}'
        else:
            ind_file = os.path.join(root, 'same_orbit_indexes.json')
            self.get_orbit_key = lambda cls, _: cls
        self.same_orbit_indexes = load_json(ind_file)[split]

        cls_file = os.path.join(root, 'cls_to_id.json')
        self.cls_to_id = load_json(cls_file)

        self._cache = \
                MultiprocessFileCache(len(self), dir=self.cache_dir, do_pickle=True)
        if os.path.exists(self.cache_path()):
            self.load_cache()

    def cache_path(self):
        return os.path.join(self.cache_dir, self.cache_file_name)
    
    def local_cache_path(self):
        return os.path.join(self.root, self.cache_file_name)

    def _load_image(self, index):
        img_path = self.meta[index][0].split('abo-material/')[-1]
        img_path = os.path.join(self.root, img_path)
        with pfio.v2.open_url(img_path, 'rb') as f:
            x = bytearray(f.read())
            x = PIL.Image.open(io.BytesIO(x)).convert('RGB')
            x = TF.resize(x, self.img_size)
        return x

    def _get_instance(self, index):
        img = self._cache.get_and_cache(index, self._load_image)
        # img = self._load_image(index)
        if self.transform is not None:
            img = self.transform(img)
        _, quat, clsname, view, env = self.meta[index]
        quat = torch.Tensor(quat)
        return img, quat, clsname, env

    def __getitem__(self, index):
        img, quat, clsname, env = self._get_instance(index)
        same_orbit_inds = self.same_orbit_indexes[self.get_orbit_key(clsname, env)]
        if self.gridsample:
            sampled_inds = same_orbit_inds
        else:
            sampled_inds = random.choices(same_orbit_inds, k=self.num_views)
        imgs = [img]
        quats = [quat]
        for ind in sampled_inds:
            img, quat, _, _ = self._get_instance(ind)
            imgs.append(img)
            quats.append(quat)
        
        cls_id = self.cls_to_id[clsname]
        return imgs, quats, cls_id

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
        dataset = ABO_Material(root, split=split, transform=trans)
        loader = DataLoader(dataset, batch_size=1024, num_workers=8)
        stats = data_stats(loader)
        print(f'mean: {[v / 255 for v in stats.mean]}')
        print(f'std: {[v / 255 for v in stats.stddev]}')
        dataset.save_cache()
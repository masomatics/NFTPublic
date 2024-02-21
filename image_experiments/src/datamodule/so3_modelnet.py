import os
import random

import torch
import torchvision

class SO3_ModelNet10(torchvision.datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, num_views=2, with_mask=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.num_views = num_views
        self.split = split

        datatype = split
        self.imgs = torch.load(os.path.join(self.root, f'{datatype}_imgs.pt'))
        #self.cats = torch.load(os.path.join(self.root, self.base_folder, f'{datatype}_categories.pt'))
        self.labels = torch.load(os.path.join(self.root, f'{datatype}_labels.pt'))
        self.quats = torch.load(os.path.join(self.root, f'{datatype}_quats.pt'))
        self.label_index = torch.load(os.path.join(self.root, f'{datatype}_label_index.pt'))
        self.len_index = torch.load(os.path.join(self.root, f'{datatype}_len_index.pt'))
        self.with_mask = with_mask

    def get_inds(self, index, num_views=None):
        num_views = self.num_views if num_views is None else num_views
        target = self.labels[index]
        inds = torch.randperm(self.len_index[target])[:(num_views - 1)]
        inds = index.tolist() + self.label_index[target][inds].tolist()
        print(inds)
        # inds = [self.label_index[target][ind] for ind in inds]
        return inds, target

    def _get_instance(self, index):
        img = self.imgs[index].div(255)
        if self.transform is not None:
            img = self.transform(img)
        quat = self.quats[index]
        return img, quat

    def __getitem__(self, index, num_views=None):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        same_orbit_inds = self.label_index[target]
        sampled_inds = random.choices(same_orbit_inds, k=(self.num_views - 1))
        img, quat = self._get_instance(index)
        imgs = [img]
        quats = [quat]
        for ind in sampled_inds:
            img, quat = self._get_instance(ind)
            imgs.append(img)
            quats.append(quat)

        # if self.with_mask:
        #     mask = (imgs != 255).all(axis=1)[..., None].to(dtype=torch.uint8) * 255
        #     target = mask
        return imgs, quats, target

    def __len__(self) -> int:
        return len(self.imgs)


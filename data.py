import re
from random import random
from pathlib import Path

from more_itertools import flatten
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from util import Normalization, dose2locs


class ImageDataset(Dataset):
    def __init__(self, folder, image_size, transparent=False, train=True, norm_f=None
                 aug_prob=0., greyscale=False, doses=[0.0], label=False):

        def paths(folder, doses):
            not_52 = re.compile('/[^(52)]')
            assays = flatten(dose2locs[dose] for dose in doses)
            gen = flatten((Path(f'{folder}').glob(
                f'**/*{assay}*.pt')) for assay in assays)
            return [p for p in gen if not_52.search(str(p))]

        self.dose2id = {k: i for i, k in enumerate(doses)}
        self.f = d8 if train else (lambda x: x)
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.label = label
        self.norm_f = norm_f or (lambda x: x)

        self.paths = paths(folder, doses)
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        #convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        self.num_channels = 3 if not transparent else 5
        self.num_channels = 1 if greyscale else self.num_channels

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomPerspective(p=aug_prob),
            transforms.RandomErasing(p=aug_prob),
            # transforms.ColorJitter(saturation=.1, contrast=.1)
            # RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = self.norm_f(torch.load(path))
        if self.num_channels == 3:
            img = img[:3]
        elif self.num_channels == 1:
            img = img[1:2]
        if self.label:
            label = self.dose2id[loc2dose[str(path).split()[-2]]]
            return self.transform(self.f(img/255)), label
        return self.transform(self.f(img/255))


class MS_Norm:  
    def __init__(self, norm_path):
        self.mean, self.std = torch.load(norm_path)
        
    def __call__(self, im):
        return (img - self.mean) / self.std


def d8(img):
    r = random()
    if r > .75:
        img = torch.rot90(img, 3, dims=(1,2))
    elif r > .5:
        img = torch.rot90(img, 2, dims=(1,2))
    elif r > .25:
        img = torch.rot90(img, 1, dims=(1,2))
    if random()>.5:
        img = torch.flip(img, dims=(2,))
    return img

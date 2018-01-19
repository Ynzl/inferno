import io
import os
import numpy as np
from os.path import join, relpath, abspath
import h5py
import torch.utils.data as data
# from PIL import Image

# from ...utils.exceptions import assert_
# from ..transform.base import Compose
# from ..transform.generic import \
#     Normalize, NormalizeRange, Cast, AsTorchBatch, Project, Label2OneHot
# from ..transform.image import \
#     RandomSizedCrop, RandomGammaCorrection, RandomFlip, Scale, PILImage2NumPyArray
# from ..core import Concatenate


class Rapoport(data.Dataset):
    splits = ('train', 'val', 'test')

    def __init__(self, root_folder, split='train',
                 image_transform=None, label_transform=None, joint_transform=None):
        """
        Parameters:
        root_folder: folder that contains 'rapoport_ground_truth_sub_tolerant.h5' and 'raw_sub.h5'.
        split: name of dataset spilt (i.e. 'train_extra', 'train', 'val' or 'test')
        """
        assert os.path.exists(root_folder)
        assert split in self.splits, str(split)


        self.root_folder = root_folder
        self.split = split

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform

        self.load_data()

    def load_data(self):
        with h5py.File(os.path.join(self.root_folder, "rapoport_ground_truth_sub_tolerant.h5"), 'r') as gt_file:
            self.gt_data = gt_file["volume"].value[...,0]
        with h5py.File(os.path.join(self.root_folder, "raw_sub.h5"), 'r') as img_file:
            self.img_data = img_file["volume/data/data"].value[...,0]

    def __getitem__(self, index):
        return self.img_data[index], self.gt_data[index]

    def __len__(self):
        return len(img_data.shape[0])


def get_rapoport_loaders(root_folder, split='all', shuffle=True, num_workers=2):
    train_set = Rapoport(root_folder, split='train')
    val_set = Rapoport(root_folder, split='val')
    test_set = Rapoport(root_folder, split='test')

    train_loader = data.DataLoader(train_set, shuffle=shuffle, num_workers=num_workers)
    val_loader = data.DataLoader(val_set, shuffle=shuffle, num_workers=num_workers)
    test_loader = data.DataLoader(test_set, shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader, test_loader

from torch.utils.data import Dataset
import h5py
import torch
from PIL import Image
import numpy as np

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class ARCPairs(Dataset):
    def __init__(self, root, phase, transform=None):
        with h5py.File(root, "r") as f:
            self.data = f[phase][()]  # (num_tasks, N_pairs, 2, H, W, C)
        self.transform = transform

    def __getitem__(self, index):
        item = self.data[index]  # (N_pairs, 2, H, W, C)
        N_pairs, two, H, W, C = item.shape
        x = torch.from_numpy(item.reshape(-1, H, W, C)).float()
        x = x.permute(0, 3, 1, 2)  # (N_pairs*2, C, H, W)
        if self.transform is not None:
            x = self.transform(x)
        x = x / 255.0
        img_size = x.shape[-1]
        return x.reshape(N_pairs, two, C, img_size, img_size)

    def __len__(self):
        return len(self.data)


class Shapes3D(Dataset):
    
    def __init__(self, root, phase,  transform=None,):
        
        with h5py.File(root, 'r') as f:
            self.imgs = f[phase][()]
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.imgs[index]
        num_relations, num_example, H, W, C = img.shape
        img = torch.from_numpy(img.reshape(-1, H, W, C))
        img = img.permute(0, -1, -3, -2)
        if self.transform is not None:
            img = self.transform(img)
        img = img.float() / 255.
        img_size = img.shape[-1]
        return img.reshape(num_relations, num_example, C, img_size, img_size)

    def __len__(self):
        return len(self.imgs)

class BitMoji(Dataset):

    def __init__(self, root, phase,  transform=None,):
       
        with h5py.File(root, 'r') as f:
            self.imgs = f[phase][()]
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.imgs[index]
        num_relations, num_example, H, W, C = img.shape
        img = torch.from_numpy(img.reshape(-1, H, W, C))
        img = img.permute(0, -1, -3, -2)
        if self.transform is not None:
            img = self.transform(img)
        img = img.float() / 255.
        img_size = img.shape[-1]
        return img.reshape(num_relations, num_example, C, img_size, img_size)

    def __len__(self):
        return len(self.imgs)

    
class CLEVr(Dataset):
    def __init__(self, root, phase,  transform=None,):
        
        with h5py.File(root, 'r') as f:
            self.imgs = f[phase][()]
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.imgs[index]
        num_relations, num_example, H, W, C = img.shape
        img = torch.from_numpy(img.reshape(-1, H, W, C))
        img = img.permute(0, -1, -3, -2)
        if self.transform is not None:
            img = self.transform(img)
        img = img.float() / 255.
        img_size = img.shape[-1]
        return img.reshape(num_relations, num_example, C, img_size, img_size)

    def __len__(self):
        return len(self.imgs)





    

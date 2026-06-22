"""
Drop-in dataset class for the OCL training loop, matching the .h5 written by
convert_to_ocl.py.

h5 layout per phase: (num_tasks, N_pairs, 2, 32, 32, 3), colours on a 0..255
scale across 3 identical channels.

This returns, per item: (N_pairs, 2, C, H, W) float in [0,1].
In train_ocl.py the DataLoader adds the batch dim -> (B, N_pairs, 2, C, H, W),
which augment_analogy and OCL.forward consume directly:
    augment_analogy unpacks (B, num_examples, _, C, H, W)  -> num_examples = N_pairs, _ = 2
    model(images[:,:-1], images[:,-1], ...)                -> support / query split

Usage in train_ocl.py:
    from create_dataset_ocl import ARCPairs
    train_dataset = ARCPairs(root=args.data_path, phase='train', transform=transform)
    val_dataset   = ARCPairs(root=args.data_path, phase='val',   transform=transform)
and launch with --image_size 32 (the dVAE downsamples by 4, so 32 -> 8x8 latent).
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class ARCPairs(Dataset):
    def __init__(self, root, phase, transform=None):
        with h5py.File(root, "r") as f:
            self.data = f[phase][()]   # (num_tasks, N_pairs, 2, H, W, C)
        self.transform = transform

    def __getitem__(self, index):
        item = self.data[index]                     # (N_pairs, 2, H, W, C)
        N_pairs, two, H, W, C = item.shape
        x = torch.from_numpy(item.reshape(-1, H, W, C)).float()
        x = x.permute(0, 3, 1, 2)                   # (N_pairs*2, C, H, W)
        if self.transform is not None:
            x = self.transform(x)
        x = x / 255.0
        img_size = x.shape[-1]
        return x.reshape(N_pairs, two, C, img_size, img_size)

    def __len__(self):
        return len(self.data)

"""
Pretrain the SLATE autoencoder on your images, then save the ENCODER weights
in the form the OCL expects (loaded into model.slate_encoder with strict=True).

Why this exists:
  The OCL (ocl.py) builds self.slate_encoder = SLATE_encoder(args) and, in
  train_ocl.py, optionally loads a pretrained encoder:
      model.slate_encoder.load_state_dict(torch.load(slate_encoder_path), strict=True)
  That checkpoint must be exactly a SLATE_encoder state_dict. In the SLATE
  autoencoder class the encoder is the submodule `self.encoder`, so we save
  slate.encoder.state_dict() (NOT slate.state_dict()).

Data:
  Uses the SAME h5 the OCL uses (produced by convert_to_ocl.py), but SLATE is a
  plain autoencoder: it trains on individual images, so we flatten the
  (N_pairs, 2, C, H, W) structure into a stream of single images.

Key detail for SLATE specifically:
  SLATE needs the Gumbel-Softmax temperature `tau` annealed from ~1.0 down to a
  small value over training, and the dVAE learning rate handled separately.
  This mirrors the original SLATE training recipe.
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('../learners/')
sys.path.append('../utils/')
from models.ocl.utils.SLATE import SLATE  # the full autoencoder (encoder + transformer decoder)


class FlatImages(Dataset):
    """
    Reads the OCL h5 (num_tasks, N_pairs, 2, H, W, C) and serves individual
    images for autoencoder training. Colours already 0..255; we scale to [0,1].
    """
    def __init__(self, root, phase):
        import h5py
        with h5py.File(root, 'r') as f:
            data = f[phase][()]                      # (T, N_pairs, 2, H, W, C)
        T, P, two, H, W, C = data.shape
        self.imgs = data.reshape(T * P * two, H, W, C)
        self.H, self.W, self.C = H, W, C

    def __getitem__(self, i):
        x = torch.from_numpy(self.imgs[i]).float()   # (H, W, C)
        x = x.permute(2, 0, 1) / 255.0               # (C, H, W)
        return x

    def __len__(self):
        return len(self.imgs)


def cosine_anneal(step, start, final, start_step, final_step):
    if step < start_step:
        return start
    if step >= final_step:
        return final
    a = 0.5 * (start - final)
    b = 0.5 * (start + final)
    progress = (step - start_step) / (final_step - start_step)
    return a * math.cos(math.pi * progress) + b


def linear_warmup(step, start, peak, start_step, peak_step):
    if step < start_step:
        return start
    if step <= peak_step:
        return (peak - start) * ((step + 1 - start_step) / (peak_step - start_step)) + start
    return peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', required=True, help='OCL h5 from convert_to_ocl.py')
    p.add_argument('--save_path', default='slate_encoder.pt.tar')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--lr_main', type=float, default=1e-4)
    p.add_argument('--lr_dvae', type=float, default=3e-4)
    p.add_argument('--lr_warmup_steps', type=int, default=30000)

    # tau (Gumbel-Softmax) annealing
    p.add_argument('--tau_start', type=float, default=1.0)
    p.add_argument('--tau_final', type=float, default=0.1)
    p.add_argument('--tau_steps', type=int, default=30000)
    p.add_argument('--hard', action='store_true')

    # must match what you will pass to the OCL
    p.add_argument('--image_size', type=int, default=32)
    p.add_argument('--vocab_size', type=int, default=128)
    p.add_argument('--d_model', type=int, default=192)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--num_dec_blocks', type=int, default=4)
    p.add_argument('--num_iterations', type=int, default=3)
    p.add_argument('--num_slots', type=int, default=3)
    p.add_argument('--num_slot_heads', type=int, default=1)
    p.add_argument('--slot_size', type=int, default=192)
    p.add_argument('--mlp_hidden_size', type=int, default=192)
    p.add_argument('--img_channels', type=int, default=3)
    p.add_argument('--pos_channels', type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_set = FlatImages(args.data_path, 'train')
    val_set = FlatImages(args.data_path, 'val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = SLATE(args).cuda()

    optimizer = Adam([
        {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
        {'params': (x[1] for x in model.named_parameters() if 'dvae' not in x[0]), 'lr': args.lr_main},
    ])

    train_epoch_size = len(train_loader)
    best_val = math.inf

    for epoch in range(args.epochs):
        model.train()
        for batch, images in enumerate(train_loader):
            step = epoch * train_epoch_size + batch
            tau = cosine_anneal(step, args.tau_start, args.tau_final, 0, args.tau_steps)
            warm = linear_warmup(step, 0., 1.0, 0, args.lr_warmup_steps)
            optimizer.param_groups[1]['lr'] = warm * args.lr_main

            images = images.cuda()
            recon, mse, ce, attns = model(images, tau, args.hard)
            loss = mse + ce

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip, 'inf')
            optimizer.step()

            if batch % max(1, train_epoch_size // 5) == 0:
                print(f'epoch {epoch+1} [{batch}/{train_epoch_size}] '
                      f'loss={loss.item():.4f} mse={mse.item():.4f} ce={ce.item():.4f} tau={tau:.3f}')

        # validation
        model.eval()
        v = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.cuda()
                recon, mse, ce, attns = model(images, args.tau_final, args.hard)
                v += (mse + ce).item()
        v /= max(1, len(val_loader))
        print(f'==> epoch {epoch+1} val_loss={v:.4f}')

        if v < best_val:
            best_val = v
            # CRITICAL: save the ENCODER submodule, which is what the OCL loads.
            torch.save(model.encoder.state_dict(), args.save_path)
            print(f'    saved encoder -> {args.save_path} (val {best_val:.4f})')


if __name__ == '__main__':
    main()
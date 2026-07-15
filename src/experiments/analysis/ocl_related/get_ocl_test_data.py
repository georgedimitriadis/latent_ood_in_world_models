import os
from os.path import join

import click
import torch
from models.ocl.learners.ocl import OCL
from dataclasses import dataclass
from models.ocl.utils.create_dataset import ARCPairs
from torchvision import transforms
from torch.utils.data import DataLoader
from models.ocl.utils.test_time_eval import snap_to_palette, build_palette_tensor
from visualization.basic_visualisation_of_data import plot_data
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

@dataclass
class Args:
    num_workers: int = 40
    seed: int = 0
    batch_size: int = 32
    epochs: int = 150
    patience: int = 4
    clip: float = 1.0
    image_size: int = 32

    dataset: str = 'arcpairs'
    checkpoint_path: str | None = r'saved_models\rotate\arcpairs\compositional_rotate_ours_0\best_model.pt'
    slate_encoder_path: str = r'saved_models\translate\slate_encoder_translate.pt.tar'
    log_path: str = r'saved_models\translate'
    data_path: str = r'data\processed\compositional_translate'

    lr_main: float = 1e-4
    lr_slate_encoder: float = 1e-5
    lr_warmup_steps: int = 15000

    num_heads: int = 4
    num_enc_heads: int = 4
    num_enc_blocks: int = 4
    num_dec_blocks: int = 4
    vocab_size: int = 128
    d_model: int = 192
    dropout: float = 0.1

    num_iterations: int = 3
    num_slots: int = 3
    num_slot_heads: int = 1
    slot_size: int = 192
    mlp_hidden_size: int = 192
    img_channels: int = 3
    pos_channels: int = 4

    tau: float = 0.1
    hard: bool = False



def get_ocl_results(saved_models_path: str,  processed_data_path: str, data_type: str = 'test'):
    device = 'cuda'

    palette = build_palette_tensor(
        [(255, 255, 255),  # white surround (code 0)
         (0, 0, 0),  # black canvas (code 1)
         (0, 116, 217),  # blue (2)
         (255, 65, 54),  # red (3)
         (255, 220, 0)],  # yellow (5)  -- include only colours that appear
        device=device)

    sampler = None

    args = Args()

    if data_type in ['train', 'val']:
        loader_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': True,
        }
    elif data_type == 'test':
        loader_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': False,
        }

    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
        ]
    )

    results = {}
    for transl_or_rot in ['translate', 'rotate']:
        print(f' Working on {transl_or_rot} data ...')
        args.checkpoint_path = f'{saved_models_path}\\{transl_or_rot}\\arcpairs\compositional_{transl_or_rot}_ours_0\\best_model.pt'
        args.slate_encoder_path = f'{saved_models_path}\\{transl_or_rot}\slate_encoder_{transl_or_rot}.pt.tar'
        args.log_path = f'{saved_models_path}\\{transl_or_rot}'
        args.data_path = f'{processed_data_path}\compositional_{transl_or_rot}'

        model = OCL(args)
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        if data_type == 'train':
            dataset = [ARCPairs(root=os.path.join(args.data_path, 'train_clo.h5'), phase='train', transform=transform)]
        elif data_type == 'val':
            dataset = [ARCPairs(root=os.path.join(args.data_path, 'train_clo.h5'), phase='val', transform=transform)]
        elif data_type == 'test':
            dataset = [
                ARCPairs(root=os.path.join(args.data_path, f'test_d{i}_clo.h5'), phase='train', transform=transform) for i in
                range(3)]
        else:
            print('data_type must be train, val or test')

        distances = [0, 1, 2] if data_type == 'test' else [0]
        data_loader = [DataLoader(dataset[i], sampler=sampler, **loader_kwargs) for i in range(len(distances))]

        print('  Collating results:')
        results[transl_or_rot] = {}
        for distance in distances:
            loader = data_loader[distance]

            support_all, query_all, gen_idx_all, target_idx_all, images_all, atten_all = [], [], [], [], [], []

            for i, images in enumerate(loader):
                print('    ', distance, i)
                images = images.to(device, non_blocking=True)
                B, N, _, C, H, W = images.shape
                # support = all pairs except the last; query input = A of the last pair
                support = images[:, :-1]  # (B, N, 2, C, H, W)
                query = images[:, -1, 0]  # (B, C, H, W)
                target = images[:, -1, 1]  # (B, C, H, W) ground-truth D

                gen = model.generate(support, query)  # (B, C, H, W) in [0,1]

                # palette-snapped exact match
                sup_temp = torch.zeros(
                    (support.shape[0], support.shape[1], support.shape[2], support.shape[4], support.shape[5]))
                for i in range(2):
                    for j in range(2):
                        sup_temp[:, i, j, :, :] = snap_to_palette(support[:, i, j, :, :, :], palette).detach().cpu()
                query = snap_to_palette(query, palette).detach().clone().cpu()
                gen_idx = snap_to_palette(gen, palette).detach().clone().cpu()  # (B, H, W)
                target_idx = snap_to_palette(target, palette).detach().clone().cpu()  # (B, H, W)

                support_all.append(sup_temp)
                query_all.append(query)
                gen_idx_all.append(gen_idx)
                target_idx_all.append(target_idx)

                _, _, _, attns = model(images[:, :-1], images[:, -1], args.tau, args.hard)
                attns_cpu = attns.detach().clone().cpu()
                attns_cpu = attns_cpu.reshape((B, 2*N, args.num_slots, C, H, W))
                atten_all.append(attns_cpu)
                images_all.append(images.detach().clone().cpu())

            results[transl_or_rot][distance] = {
                'support': torch.cat(support_all, dim=0).numpy(),  # (B=999, N-1=2, 2(each pair), C=3, H=32, W=32)
                'query': torch.cat(query_all, dim=0).numpy(),  # (B, C, H, W)
                'gen_idx': torch.cat(gen_idx_all, dim=0).numpy(),  # (B, H, W)
                'target_idx': torch.cat(target_idx_all, dim=0).numpy(),  # (B, H, W)
                'images': torch.cat(images_all, dim=0).numpy(), # (B, N=3(# of pairs), 2(each pair), C, H, W)
                'attention_slots': torch.cat(atten_all, dim=0).numpy() # (B, N * 2(each pair), num_slots, C, H, W)
            }

    return results
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



@click.command()
@click.argument('saved_models_path', default='saved_models', type=click.Path())
@click.argument('processed_data_path', default='data\processed', type=click.Path())
def main(saved_models_path: str,  processed_data_path: str):
    device = 'cuda'

    palette = build_palette_tensor(
        [(255, 255, 255),  # white surround (code 0)
         (0, 0, 0),  # black canvas (code 1)
         (0, 116, 217),  # blue (2)
         (255, 65, 54),  # red (3)
         (255, 220, 0)],  # yellow (5)  -- include only colours that appear
        device=device)

    train_sampler = None
    val_sampler = None
    test_sampler = None

    args = Args()

    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': True,
    }

    test_loader_kwargs = {
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

        #train_dataset = ARCPairs(root=os.path.join(args.data_path, 'train_clo.h5'), phase='train', transform=transform)
        #val_dataset = ARCPairs(root=os.path.join(args.data_path, 'train_clo.h5'), phase='val', transform=transform)
        test_datasets = [
            ARCPairs(root=os.path.join(args.data_path, f'test_d{i}_clo.h5'), phase='train', transform=transform) for i in
            range(3)]

        #train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
        #val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
        test_loaders = [DataLoader(test_datasets[i], sampler=test_sampler, **test_loader_kwargs) for i in range(3)]

        print('  Collating results:')
        results = {}
        for distance in [0, 1, 2]:
            test_loader = test_loaders[distance]

            support_all, query_all, gen_idx_all, target_idx_all = [], [], [], []

            for i, images in enumerate(test_loader):
                print('    ', distance, i)
                images = images.to(device, non_blocking=True)
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

            results[distance] = {
                'support': torch.cat(support_all, dim=0).numpy(),  # (999, N-1, 2, C, H, W)
                'query': torch.cat(query_all, dim=0).numpy(),  # (999, C, H, W)
                'gen_idx': torch.cat(gen_idx_all, dim=0).numpy(),  # (999, H, W)
                'target_idx': torch.cat(target_idx_all, dim=0).numpy(),  # (999, H, W)
            }

        save_figure_path = f'data/results/{transl_or_rot}/figures/ocl'
        if not os.path.exists(save_figure_path):
            Path(save_figure_path).mkdir(parents=True, exist_ok=True)

        print('  Saving images:')
        for distance in [0, 1, 2]:
            for batch_idx in range(999):
                print('    ', distance, batch_idx)
                fig, ax = plt.subplots(4, 2, figsize=(13, 19.5))
                extent = [-0.5, 31.5, -0.5, 31.5]
                plot_data(results[distance]['support'][batch_idx][0][0], extent, ax[0, 0])
                plot_data(results[distance]['support'][batch_idx][0][1], extent, ax[0, 1])
                plot_data(results[distance]['support'][batch_idx][1][0], extent, ax[1, 0])
                plot_data(results[distance]['support'][batch_idx][1][1], extent, ax[1, 1])
                plot_data(results[distance]['query'][batch_idx], extent, ax[2, 0])
                plot_data(results[distance]['target_idx'][batch_idx], extent, ax[2, 1])
                plot_data(results[distance]['gen_idx'][batch_idx], extent, ax[3, 1])
                ax[3, 0].axis('off')
                plt.tight_layout()

                save_figure_filename = join(save_figure_path, f'dist_{distance}__im_{batch_idx}')
                fig.savefig(f'{save_figure_filename}.png')
                fig.savefig(f'{save_figure_filename}.svg')
                plt.close(fig)

if __name__ == '__main__':
    main()
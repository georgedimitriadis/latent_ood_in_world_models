import os
from os.path import join

import click
from visualization.basic_visualisation_of_data import plot_data
from pathlib import Path
from matplotlib import pyplot as plt
from experiments.analysis.ocl_related.get_ocl_test_data import get_ocl_results


@click.command()
@click.argument('saved_models_path', default='saved_models', type=click.Path())
@click.argument('processed_data_path', default='data\processed', type=click.Path())
def main(saved_models_path: str,  processed_data_path: str):

    # Get all the data from the three test distances (the two analogies, the input, the trained output and the model's prediction
    all_results = get_ocl_results(saved_models_path, processed_data_path, data_type='test')

    for transl_or_rot in ['translate', 'rotate']:
        results = all_results[transl_or_rot]

        save_figure_path = f'data/results/{transl_or_rot}/figures/ocl'
        if not os.path.exists(save_figure_path):
            Path(save_figure_path).mkdir(parents=True, exist_ok=True)

        print(f'Saving images for {transl_or_rot}:')
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
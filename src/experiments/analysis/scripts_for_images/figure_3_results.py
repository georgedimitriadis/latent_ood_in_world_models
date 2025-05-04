
import os
from pathlib import Path
from os.path import join
import click
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st


@click.command()
@click.argument('results_path', type=click.Path())
@click.argument('save_figure_path', type=click.Path())
@click.argument('iterations', type=click.INT)
def main(results_path: str, save_figure_path: str, iterations: int):
    data_types = ['translate', 'rotate']
    model_types = ['axial_pointer_network_lines', 'axial_pointer_network_full', 'cnn', 'mlp_nn', 'transformer']
    model_names = ['Axial Pointer Linear', 'Axial Pointer', 'CNN', 'MLP', 'Transformer']
    num_of_distances = 3

    # Find number of epochs
    f = join(results_path, 'translate', 'logs', f'run_{0}', f'log_translate_axial_pointer_network_lines.csv')
    raw_results = pd.read_csv(f, usecols=[1, 4], skiprows=[0, 0], names=['error', 'distance'])
    num_of_epochs = len(raw_results) // num_of_distances

    all_results = {}
    for data_type in data_types:
        all_results[data_type] = {0: [], 1: [], 2: []}
        for model in model_types:
            model_iteration_results = np.zeros((num_of_epochs, iterations, 3))
            for iteration in range(iterations):
                f = join(results_path, data_type, 'logs', f'run_{iteration}', f'log_{data_type}_{model}.csv')
                raw_results = pd.read_csv(f, usecols=[1, 4], skiprows=[0, 0], names=['error', 'distance'])
                for d in range(num_of_distances):
                    model_iteration_results[:, iteration, d] = raw_results['error'][raw_results['distance'] == d].to_numpy()
            for d in range(num_of_distances):
                all_results[data_type][d].append(model_iteration_results[:, :, d])

    fig = plt.figure(constrained_layout=False, figsize=(15, 10), dpi=50)
    subfigs = fig.subfigures(nrows=2, ncols=1)

    for data_type, subfig in zip(data_types, subfigs):
        subfig.suptitle(data_type)
        epochs = range(num_of_epochs)
        axs = subfig.subplots(nrows=1, ncols=3)
        for d, ax in zip([0, 1, 2], axs):
            lines = []
            for m in range(len(model_types)):
                data = all_results[data_type][d][m]
                loc = np.mean(data, axis=1)
                scale = st.sem(data, axis=1)
                mph, mmh = st.t.interval(0.95, len(np.transpose(data)) - 1, loc=loc, scale=scale)
                lines.append(ax.plot(epochs, loc, label=model_names[m])[0])
                ax.fill_between(range(num_of_epochs), mmh, mph, color='b', alpha=.1)
                ax.set_title(f'distance {d}')
            if data_type == 'translate' and d == 2:
                ax.legend(handles=lines, loc='center right')
    fig.text(0.5, 0.01, 'epochs', ha='center', va='center')
    fig.text(0.01, 0.5, '% error', ha='center', va='center', rotation='vertical')

    if not os.path.exists(save_figure_path):
        Path(save_figure_path).mkdir(parents=True, exist_ok=True)

    save_figure_filename = join(save_figure_path, 'learning_error_graph')
    fig.savefig(f'{save_figure_filename}.png')
    fig.savefig(f'{save_figure_filename}.svg')


if __name__ == '__main__':
    main()







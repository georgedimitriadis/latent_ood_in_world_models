import os
from os.path import join
from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt
from src.experiments.analysis.tensorboard_events_reader import get_dfs_from_events
from experiments.analysis.scripts_for_images.figure_3_results import collect_data_results

@click.command()
@click.argument('translate_event_file', type=click.Path())
@click.argument('rotate_event_file', type=click.Path())
@click.argument('results_path', type=click.Path())
@click.argument('save_figure_path', type=click.Path())
@click.argument('iterations', type=click.INT)
def main(translate_event_file: str, rotate_event_file: str, results_path:str, save_figure_path: str, iterations: int = 8):
    #translate_event_file = r'E:\Projects Large\Learning\Papers_Proposals\2026_IEEE_TCDS_Compositionality_NRIPSRedo\code\latent_ood_in_world_models\data\results\translate\arcpairs\compositional_translate_ours_0\events.out.tfevents.1782296422.cseeblg2.1402481.0'
    df_train_translate, df_val_translate, df_test_translate = get_dfs_from_events(translate_event_file)
    #rotate_event_file = r'E:\Projects Large\Learning\Papers_Proposals\2026_IEEE_TCDS_Compositionality_NRIPSRedo\code\latent_ood_in_world_models\data\results\rotate\arcpairs\compositional_rotate_ours_0\events.out.tfevents.1782296788.cseeblg2.1408796.0'
    df_train_rotate, df_val_rotate, df_test_rotate = get_dfs_from_events(rotate_event_file)
    df_test_data = [df_test_translate, df_test_rotate]

    #results_path = r'E:\Projects Large\Learning\Papers_Proposals\2026_IEEE_TCDS_Compositionality_NRIPSRedo\code\latent_ood_in_world_models\data\results'
    data_types = ['translate', 'rotate']
    model_types = ['axial_pointer_network_lines']
    all_results, num_of_epochs = collect_data_results(results_path, num_of_distances=3, data_types=data_types, model_types=model_types, iterations=iterations)

    avg_apnl_translate_results = [np.mean(all_results['translate'][d][0], axis=1) for d in range(3)]
    avg_apnl_results = [[np.mean(all_results['translate'][d][0], axis=1) for d in range(3)], [np.mean(all_results['rotate'][d][0], axis=1) for d in range(3)]]


    fig_translate, ax = plt.subplots(nrows=1, ncols=3)
    ax_twins = [plt.twinx(ax[c]) for c in range(3)]
    r = 0 # Translate
    for c in range(3):
        ax[c].plot(df_test_data[r][f'd{c}_exact_match'], label='OCL % images correct')
        ax_twins[c].plot(df_test_data[r][f'd{c}_mse'], label='OCL MSE', color='r')
        ax[c].plot(avg_apnl_results[r][c], label='APNL % images correct')
        ax[c].legend()
        ax_twins[c].legend()

    fig_rotate, ax = plt.subplots(nrows=1, ncols=3)
    ax_twins = [plt.twinx(ax[c]) for c in range(3)]
    r = 1 # Rotate
    for c in range(3):
        ax[c].plot(df_test_data[r][f'd{c}_exact_match'], label='OCL % images correct')
        ax_twins[c].plot(df_test_data[r][f'd{c}_mse'], label='OCL MSE', color='r')
        ax[c].plot(avg_apnl_results[r][c], label='APNL % images correct')
        ax[c].legend()
        ax_twins[c].legend()

    if not os.path.exists(save_figure_path):
        Path(save_figure_path).mkdir(parents=True, exist_ok=True)

    save_figure_filename = join(save_figure_path, 'ocl_learning_error_translate_graph')
    fig_translate.savefig(f'{save_figure_filename}.png')
    fig_translate.savefig(f'{save_figure_filename}.svg')

    save_figure_filename = join(save_figure_path, 'ocl_learning_error_rotate_graph')
    fig_rotate.savefig(f'{save_figure_filename}.png')
    fig_rotate.savefig(f'{save_figure_filename}.svg')

if __name__ == '__main__':
    main()
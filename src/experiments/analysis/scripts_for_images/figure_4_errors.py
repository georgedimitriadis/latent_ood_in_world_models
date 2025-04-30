
from os.path import join

import click
import keras
import numpy as np
import matplotlib.pyplot as plt

from experiments.analysis.intermediate_layer_analysis_functions import load_data
from visualization.basic_visualisation_of_data import plot_data
from models.lm import b_acc_s

'''
translate_data_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\data\object_compositionality\symmetric_translate_withoutpi'
rotate_data_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\data\object_compositionality\nonsymmetric_rotate_withoutpi'

translate_model_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\models\object_compositionality\symmetric_translate_withoutpi'
rotate_model_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\models\object_compositionality\nonsymmetric_rotate_withoutpi'

output_figures_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\figures\errors'


'''
all_models = ['axial_point_network_full', 'axial_point_network_lines', 'cnn', 'transformer', 'mlp_nn']


def generate_figure_all_models_error_example(model_filepath: str, test_data_filepath: str, save_figure_filepath: str,
                                             data_type: str, dist: int, index: int):
    fig, ax = plt.subplots(3, 3, figsize=(32, 8))
    extent = [-0.5, 31.5, -0.5, 31.5]

    title_properties = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'bold',
        'size': 12,
    }

    for i, model_name in enumerate(all_models):
        current_model_filepath = join(model_filepath, f'{model_name}.keras')
        model = keras.models.load_model(f"{current_model_filepath}")

        test_data_filepath_dist = join(test_data_filepath, f'test_d{dist}.npz')
        input_images, input_language, output_images = load_data(test_data_filepath_dist)
        input_image = np.expand_dims(input_images[index], axis=0)
        input_action = np.expand_dims(input_language[index], axis=0)
        predicted_image = model.predict(x=[input_image, input_action])

        if i == 0:
            ax[0, 0] = plot_data(input_image[0], extent, axis=ax[0, 0])
            ax[0, 0].axis('off')
            ax[0, 0].set_title('Input', fontdict=title_properties)

            action = f'Distance {dist}\n'
            if data_type == 'translate':
                action += 'Translate Up' if input_action[0] == 0 else 'Translate Left'
            if data_type == 'rotate':
                action += 'Rotate 90' if input_action[0] == 0 else 'Rotate 180'

            ax[0, 1].text(0.5, 0.5, action, fontsize=16, ha='center', va='center')
            ax[0, 1].axis('off')
            ax[0, 1].set_title("Action", fontdict=title_properties)

            ax[0, 2] = plot_data(output_images[index], extent, axis=ax[0, 2])
            ax[0, 2].axis('off')
            ax[0, 2].set_title('Output', fontdict=title_properties)

        row = i // 3 + 1
        column = i % 3
        ax[row, column] = plot_data(np.argmax(predicted_image[0], axis=-1), extent, axis=ax[row, column])
        ax[row, column].axis('off')
        ax[row, column].set_title(model_name, fontdict=title_properties)

    ax[2, 2].remove()
    plt.tight_layout()

    save_figure_filename = join(save_figure_filepath, f'errors_{data_type}_distance_{dist}_sample_{index}')
    fig.savefig(f'{save_figure_filename}.png')
    fig.savefig(f'{save_figure_filename}.svg')


@click.command()
@click.argument('data_type', type=click.STRING)
@click.argument('dist', type=click.INT)
@click.argument('index', type=click.INT)
@click.argument('model_folder', type=click.Path())
@click.argument('data_folder', type=click.Path())
@click.argument('save_figure_folder', type=click.Path())
def main(model_folder: str, data_folder: str, save_figure_folder: str, data_type: str, dist: int, index: int):

    translate_data_folder = join(data_folder, 'compositional_translate')
    rotate_data_folder = join(data_folder, 'compositional_rotate')
    translate_model_folder = join(model_folder, 'translate')
    rotate_model_folder = join(model_folder, 'rotate')

    test_data_filepath = translate_data_folder if data_type == 'translate' else rotate_data_folder
    model_filepath = translate_model_folder if data_type == 'translate' else rotate_model_folder


if __name__ == '__main__':
    main()

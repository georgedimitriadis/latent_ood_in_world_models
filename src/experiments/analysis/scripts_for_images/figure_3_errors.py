
from os.path import join

import keras
import numpy as np
import matplotlib.pyplot as plt

from experiments.analysis.intermediate_layer_analysis_functions import load_data, load_models_layers, un_one_hot
from visualization.visualize_data import plot_data

translate_data_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\data\object_compositionality\symmetric_translate_withoutpi'
rotate_data_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\data\object_compositionality\nonsymmetric_rotate_withoutpi'

translate_model_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\models\object_compositionality\symmetric_translate_withoutpi'
rotate_model_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\models\object_compositionality\nonsymmetric_rotate_withoutpi'

output_figures_folder = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\figures\errors'

all_models = ['axial_point_network_full', 'axial_point_network_lines', 'pure_cnn', 'transformer', 'mlp_nn']


def create_and_save_figs(model, X, Z, Y, logs_filepath, distance):
    Y_prime = model.predict(x=[X, Z])
    for j, (x, a, y, y_prime) in enumerate(list(zip(X, Z, Y, Y_prime))[:100]):
        y_prime = np.argmax(y_prime, axis=-1)
        filename_id = f"{logs_filepath}/distance_{distance}_{j}_{model.name}"

        fig, ax = plt.subplots(1, 4, figsize=(32, 8))

        extent = [-0.5, 31.5, -0.5, 31.5]

        # Plot the first image (x)
        ax[0] = plot_data(x, extent=extent, axis=ax[0])
        ax[0].axis('off')
        ax[0].set_title("Image X")

        # Display the action in the second position
        ax[1].text(0.5, 0.5, str(a), fontsize=16, ha='center', va='center')
        ax[1].axis('off')
        ax[1].set_title("Action")

        # Plot the second image (y)
        ax[2] = plot_data(y, extent=extent, axis=ax[2])
        ax[2].axis('off')
        ax[2].set_title("Image Y")

        # Plot the third image (y')
        ax[3] = plot_data(y_prime, extent=extent, axis=ax[3])
        ax[3].axis('off')
        ax[3].set_title("Image Y'")

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"{filename_id}.svg", bbox_inches='tight')
        plt.savefig(f"{filename_id}.png", bbox_inches='tight')
        plt.close()


def generate_figures_in_action_out_predicted_for_model_and_datatype(model_type, model_filepath, test_data_filepath, output_figures_folder):
    """
    Call the training of the compositional model
    :param output_figures_folder: The base folder where the model folders with the resulting figures are.
    :param model_type: The model type string. Can be axial_point_network_lines, axial_point_network_full, pure_cnn, transformer, mlp_nn
    :param model_filepath: The path where the model will be loaded from as output_filepath/model_type.keras
    :param test_data_filepath: The file path of the test data sets. The full paths are test_data_filepath/test_d{i}.npz since it assumes that the names of the test sets are test_d0.npz, test_d1.npz and test_d2.npz
    :return:
    """

    model_filepath = join(model_filepath, f'{model_type}.keras')
    model = keras.models.load_model(f"{model_filepath}")

    save_figures_filepath = join(output_figures_folder, model_type,
                                 'translate' if 'translate' in test_data_filepath else 'rotate')
    for dist in [0, 1, 2]:
        test_data_filepath_dist = join(test_data_filepath, f'test_d{dist}.npz')
        input_images, input_language, output_images = load_data(test_data_filepath_dist)
        create_and_save_figs(model=model, X=input_images, Z=input_language, Y=output_images,
                             logs_filepath=save_figures_filepath, distance=dist)


def generate_figures_in_out_action_predicted_for_all_models_and_datatypes():
    for model in all_models:
        for model_filepath, test_data_filepath in zip([translate_model_folder, rotate_model_folder],
                                                      [translate_data_folder, rotate_data_folder]):
            generate_figures_in_action_out_predicted_for_model_and_datatype(model_type=model,
                                                                            model_filepath=translate_model_folder,
                                                                            test_data_filepath=translate_data_folder,
                                                                            output_figures_folder=output_figures_folder)
            print(f'Finished {model} for {"translate" if "translate" in test_data_filepath else "rotate"}')


def generate_figure_all_models_error_example(type, dist, index):
    test_data_filepath = translate_data_folder if type == 'translate' else rotate_data_folder
    model_filepath = translate_model_folder if type == 'translate' else rotate_model_folder

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
            ax[0, 0].set_title('Input',  fontdict=title_properties)

            action = f'Distance {dist}\n'
            if type == 'translate':
                action += 'Translate Up' if input_action[0] == 0 else 'Translate Left'
            if type == 'rotate':
                action += 'Rotate 90' if input_action[0] == 0 else 'Rotate 180'

            ax[0, 1].text(0.5, 0.5, action, fontsize=16, ha='center', va='center')
            ax[0, 1].axis('off')
            ax[0, 1].set_title("Action",  fontdict=title_properties)

            ax[0, 2] = plot_data(output_images[index], extent, axis=ax[0, 2])
            ax[0, 2].axis('off')
            ax[0, 2].set_title('Output',  fontdict=title_properties)

        row = i//3 + 1
        column = i % 3
        ax[row, column] = plot_data(np.argmax(predicted_image[0], axis=-1), extent, axis=ax[row, column])
        ax[row, column].axis('off')
        ax[row, column].set_title(model_name,  fontdict=title_properties)

    ax[2, 2].remove()
    plt.tight_layout()
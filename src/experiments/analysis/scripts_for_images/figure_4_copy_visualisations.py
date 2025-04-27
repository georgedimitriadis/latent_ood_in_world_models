

import os
os.environ['KERAS_BACKEND'] ='jax'
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

import visualization.visualise_NN_layers as vis_layers
import experiments.analysis.intermediate_layer_analysis_functions as nn_funcs


model_base_path = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\models\object_compositionality'
data_base_path = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\data\object_compositionality'
vis_arrays_output_path = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\analysis\visualisations'
vis_figures_output_path = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\figures\visualisations'

translate_path = 'symmetric_translate_withoutpi'
rotate_path = 'nonsymmetric_rotate_withoutpi'

models = ['axial_point_network_lines', 'axial_point_network_full']
final_layers = ['attention_mapping_layer2d', 'spatial_copy_layer']

distance = 0
type = 'translate'
model_index = 0


def generate_visualisation_arrays(distance, type, model_index):
    model = models[model_index]
    final_layer = final_layers[model_index]

    data_type_path = translate_path if type == 'translate' else rotate_path
    model_filepath = join(model_base_path, data_type_path, f'{model}.keras')
    data_filepath = join(data_base_path, data_type_path, f'test_d{distance}.npz')

    X, Z, Y = nn_funcs.load_data(data_filepath)

    output_layers = nn_funcs.load_models_layers(model_filepath=model_filepath, data_filepath=data_filepath,
                                                selected_layers=['attention_logits', final_layer])

    attention_logits = output_layers[0]

    pixel_to_pixel_func = nn_funcs.pixel_to_pixel_from_hw_attention_logits_matrix_based if model_index == 0 else \
        nn_funcs.pixel_to_pixel_from_full_attention_logits_matrix_based

    visualisations_in, visualisations_out, copied_from_pixel_indices_all_images = \
        pixel_to_pixel_func(input_images=X, attention_logits=attention_logits, examples_from_batch=list(range(100)))

    vis_file = join(vis_arrays_output_path, models[model_index], type, f'visualisation_arrays_d{distance}.npz')
    np.savez(vis_file, visualisations_in=visualisations_in, visualisations_out=visualisations_out,
             copied_from_pixel_indices_all_images=copied_from_pixel_indices_all_images)


def generate_vis_arrays_for_all_distances_data_and_models():
    for distance in [0, 1, 2]:
        for type in ['translate', 'rotate']:
            for model_index in [0, 1]:
                generate_visualisation_arrays(distance, type, model_index)
                print(f'Finished {models[model_index]}, {type}, {distance}')


def load_visualisation_array(model_index, type, distance):
    vis_file = join(vis_arrays_output_path, models[model_index], type, f'visualisation_arrays_d{distance}.npz')
    temp_npz = np.load(vis_file)
    visualisations_in = temp_npz[temp_npz.files[0]]
    visualisations_out = temp_npz[temp_npz.files[1]]
    copied_from_pixel_indices_all_images = temp_npz[temp_npz.files[2]]

    return visualisations_in, visualisations_out, copied_from_pixel_indices_all_images


def generate_visualisation_figure(model_index, type, distance, image_index):
    visualisations_in, visualisations_out, copied_from_pixel_indices_all_images = load_visualisation_array(model_index,
                                                                                                           type,
                                                                                                           distance)
    fig, ax_input, ax_output = vis_layers.visualise_attention_results_with_colours(visualisations_in,
                                                                                   visualisations_out, image_index)
    fig.set_size_inches(20, 12)
    fig.suptitle(f'Model = {models[model_index]}, Data =  {type}, Distance {distance}')
    plt.tight_layout()


def generate_visualisation_figures_for_all_images():
    for distance in [0, 1, 2]:
        for type in ['translate', 'rotate']:
            for model_index in [0, 1]:
                for image_index in range(100):
                    generate_visualisation_figure(model_index, type, distance, image_index)
                    filename = join(vis_figures_output_path, models[model_index], type,
                                    f'Visualisation__{models[model_index]}__{type}__Dist_{distance}__Image_{image_index}')
                    plt.tight_layout()
                    plt.savefig(f"{filename}.svg", bbox_inches='tight')
                    plt.savefig(f"{filename}.png", bbox_inches='tight')
                    plt.close()

                print(f'        Done model {models[model_index]}')
            print(f'    Done data {type}')
        print(f'Done distance {distance}')


# line_translate_0, line_translate_1, line_translate_2
correct = {'lines': {'translate': {0: 45, 1: 1, 2: 92}, 'rotate': {0: 4, 1: 3, 2: 7}},
           'full': {'translate': {0: 17, 1: 8, 2: 1}, 'rotate': {0: 5, 1: 3, 2: None}}}

errors = {'lines': {'translate': {0: None, 1: None, 2: None}, 'rotate': {0: 13, 1: 2, 2: 2}},
          'full': {'translate': {0: None, 1: 15, 2: 0}, 'rotate': {0: 36, 1: 13, 2: 1}}}



import numpy as np
from numpy._typing import NDArray

import src.experiments.analysis.scripts_for_images.figure_5_copy_visualisations as f5
import experiments.analysis.intermediate_layer_analysis_functions as nn_funcs
from os.path import join

translate_path = f5.translate_path
rotate_path = f5.rotate_path
model = f5.models
final_layer = f5.final_layers

num_of_samples = 100
model_base_path = 'saved_models'
data_base_path = 'data/processed'
output_path = vis_arrays_output_path = 'data/results'

model_index = 0 # 0 = Lines 1 = Full
data_type = 'rotate'
distance = 0

data_type_path = translate_path if data_type == 'translate' else rotate_path
model_type_path = 'translate' if data_type == 'translate' else 'rotate'
model_filepath = join(model_base_path, model_type_path, f'{model}.keras')
data_filepath = join(data_base_path, data_type_path, f'test_d{distance}.npz')

X, Z, Y = nn_funcs.load_data(data_filepath)

visualisations_in, visualisations_out, copied_from_pixel_indices_all_images = f5.load_visualisation_array(num_of_samples,
                                                                                                           model_base_path,
                                                                                                           data_base_path,
                                                                                                           vis_arrays_output_path,
                                                                                                           model_index,
                                                                                                           data_type,
                                                                                                           distance)

image_index = 0
object_colour = np.max(X[image_index])
black_pixel_set_before_move = set([(a[1], a[2]) for a in np.argwhere(X<2) if a[0]==image_index])
object_pixel_set_before_move = set([(a[1], a[2]) for a in np.argwhere(X==object_colour) if a[0]==image_index])
black_pixel_set_after_move = set([(a[1], a[2]) for a in np.argwhere(Y<2) if a[0]==image_index])
object_pixel_set_after_move = set([(a[1], a[2]) for a in np.argwhere(Y==object_colour) if a[0]==image_index])

def get_bounding_box(object_pixels: NDArray[int]) -> NDArray[int]:
    tt = object_pixels[np.argwhere(object_pixels[:, 0] == object_pixels.min(axis=0)[0]), :].squeeze()
    bottom_left = np.array([tt[0, 0], np.min(tt[:, 1])])
    tt = object_pixels[np.argwhere(object_pixels[:, 0] == object_pixels.max(axis=0)[0]), :].squeeze()
    top_right = np.array([tt[0, 0], np.max(tt[:, 1])])

    return np.array([bottom_left, top_right])

image_index = 0
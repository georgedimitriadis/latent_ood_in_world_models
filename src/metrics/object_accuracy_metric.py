

import matplotlib.pyplot as plt
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

model_index = 1  # 0 = Lines 1 = Full
data_type = 'rotate'
distance = 1

data_type_path = translate_path if data_type == 'translate' else rotate_path
model_type_path = 'translate' if data_type == 'translate' else 'rotate'
model_filepath = join(model_base_path, model_type_path, f'{model}.keras')
data_filepath = join(data_base_path, data_type_path, f'test_d{distance}.npz')

X, Z, Y = nn_funcs.load_data(data_filepath)

visualisations_in, visualisations_out, copied_from_pixel_indices_all_images = f5.load_visualisation_array(
    num_of_samples,
    model_base_path,
    data_base_path,
    vis_arrays_output_path,
    model_index,
    data_type,
    distance)


def get_bounding_box_for_pixels(object_pixels: NDArray[int]) -> NDArray[int]:
    bottom_left = object_pixels.min(axis=0)
    top_right = object_pixels.max(axis=0)

    return np.array([bottom_left, top_right])

def get_bounding_boxes_for_X_Y(X, Y, object_colour, image_index):
    pixels_before_move = np.array(
        [(a[1], a[2]) for a in np.argwhere(X == object_colour) if a[0] == image_index])
    bounding_box_before_move = get_bounding_box_for_pixels(pixels_before_move)
    pixels_after_move = np.array([(a[1], a[2]) for a in np.argwhere(Y == object_colour) if a[0] == image_index])
    bounding_box_after_move = get_bounding_box_for_pixels(pixels_after_move)
    full_bounding_box = get_bounding_box_for_pixels(
        np.concatenate((bounding_box_before_move, bounding_box_after_move), axis=0))

    return bounding_box_before_move, bounding_box_after_move, full_bounding_box


def get_non_compositional_errors(data_type, distance, X, Y, Z, copied_from_pixel_indices_all_images):
    images_where_there_is_no_object_in_Y = {'translate': {0: [62, 94], 1: [12, 23, 31, 59, 72], 2: []},
                                            'rotate': {0: [], 1: [], 2: []}}
    random_copied_from_pixels_indices = np.random.randint(low=0, high=31,
                                                                  size=copied_from_pixel_indices_all_images.shape)
    object_pixels_errors = []
    other_pixels_errors = []
    sum_of_errors = []
    image_indices = []

    for image_index in range(num_of_samples):

        object_colour = np.max(X[image_index])

        bounding_box_before_move, bounding_box_after_move, full_bounding_box = (
            get_bounding_boxes_for_X_Y(X, Y, object_colour, image_index))

        distance_to_correct_copy = 0

        if data_type == 'translate' and image_index not in images_where_there_is_no_object_in_Y[data_type][distance]:
            z = Z[image_index]
            object_pixels_translation = [0, 6] if z == 1 else [6, 0]

            pixels_after_move = np.array([(a[1], a[2]) for a in np.argwhere(Y == object_colour) if a[0] == image_index])
            for to_pixel in pixels_after_move:
                from_pixel = np.array([copied_from_pixel_indices_all_images[image_index, to_pixel[0], to_pixel[1], 0],
                                       copied_from_pixel_indices_all_images[image_index, to_pixel[0], to_pixel[1], 1]])
                dist_moved = np.abs(from_pixel - to_pixel)
                distance_to_correct_copy += np.sqrt(np.sum(np.power(dist_moved - object_pixels_translation, 2)))

        if data_type == 'rotate':
            for to_x in range(bounding_box_after_move[0, 1], bounding_box_after_move[1, 1] + 1, 1):
                for to_y in range(bounding_box_after_move[0, 0], bounding_box_after_move[1, 0] + 1, 1):
                    if Y[image_index, to_y, to_x] == object_colour:
                        from_pixel = np.array([copied_from_pixel_indices_all_images[image_index, to_y, to_x, 0],
                                               copied_from_pixel_indices_all_images[image_index, to_y, to_x, 1]], dtype=int)
                        if X[image_index, from_pixel[0], from_pixel[1]] == object_colour:
                            pass
                        else:
                            to_pixel = np.array([to_y, to_x])
                            distance_to_correct_copy += np.sqrt(np.sum(np.power(from_pixel - to_pixel, 2)))

        num_of_pixels_in_full_bb = 0
        pixel_copy_of_other_pixels = np.copy(copied_from_pixel_indices_all_images[image_index, :, :])
        for x in range(full_bounding_box[0, 1], full_bounding_box[1, 1] + 1, 1):
            for y in range(full_bounding_box[0, 0], full_bounding_box[1, 0] + 1, 1):
                pixel_copy_of_other_pixels[y, x, 0] = y
                pixel_copy_of_other_pixels[y, x, 1] = x
                num_of_pixels_in_full_bb += 1
        distance_to_correct_copy /= num_of_pixels_in_full_bb

        correct_copy_of_other_pixels = np.zeros((32, 32, 2))
        for x in range(32):
            for y in range(32):
                correct_copy_of_other_pixels[y, x, 0] = y
                correct_copy_of_other_pixels[y, x, 1] = x

        other_pixels_error = (np.sum(
                                 np.sqrt(
                                    np.sum(
                                        np.power(pixel_copy_of_other_pixels - correct_copy_of_other_pixels, 2),
                                        axis=2)
                                    )
                              ) /
                              (1024 - num_of_pixels_in_full_bb))

        object_pixels_errors.append(distance_to_correct_copy)
        other_pixels_errors.append(other_pixels_error)
        sum_of_errors.append(distance_to_correct_copy + other_pixels_error)
        image_indices.append(image_index)

    random_error = np.sum(
                        np.sqrt(
                            np.sum(
                                np.power(copied_from_pixel_indices_all_images - random_copied_from_pixels_indices, 2),
                                axis=3)
                        ),
                        axis=(1,2)) / 1024

    return (np.array(image_indices), np.array(sum_of_errors), np.array(object_pixels_errors),
            np.array(other_pixels_errors), random_error)




image_indices, sum_of_errors, object_pixels_errors, other_pixels_errors, random_error = get_non_compositional_errors(data_type,
                                                                                                       distance, X, Y,
                                                                                                       Z,
                                                                                                       copied_from_pixel_indices_all_images)

plt.plot(image_indices, object_pixels_errors, image_indices, other_pixels_errors, image_indices, random_error[image_indices])
plt.legend(('object', 'other', 'random'))

all_errors = {'Non object pixels copying' : other_pixels_errors, 'Object pixels copying': object_pixels_errors}
fig, ax = plt.subplots()
bottom = np.zeros(image_indices.shape)
w = -0.5

for name, errors in all_errors.items():
    p = ax.bar(x=image_indices, height=errors, width=1, label=name, bottom=bottom)
    ax.line(x=image_indices, y=errors)
    bottom += errors


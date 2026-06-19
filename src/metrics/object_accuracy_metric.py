

import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import NDArray

import src.experiments.analysis.scripts_for_images.figure_5_copy_visualisations as f5
import experiments.analysis.intermediate_layer_analysis_functions as nn_funcs
from os.path import join

translate_path = f5.translate_path
rotate_path = f5.rotate_path
final_layer = f5.final_layers

def load_data(model_index, data_type, distance):
    num_of_samples = 100
    model_base_path = 'saved_models'
    data_base_path = 'data/processed'

    data_type_path = translate_path if data_type == 'translate' else rotate_path
    vis_arrays_output_path = 'data/results'
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

    return X, Z, Y, copied_from_pixel_indices_all_images


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
    """
    Calculates the root mean (per pixel) square error between the distance a pixel has moved from X to Y_hat and the
    distance it should have moved if the APNs had figured out the object in the image and moved only that appropriately
    :param data_type: 'rotate' or 'translate'
    :param distance: 0, 1, 2 the OOD distance
    :param X: The input images to the APN model
    :param Y: The correct output of the transformation
    :param Z: The transformation (0 or 1)
    :param copied_from_pixel_indices_all_images: The result of the final layer of the APNs. It shows (for each x and y)
                                                 which pixel from x was copied to the y_hat image.
    :return: 1) the indices of the images (not all images generate an error). 2) The rmse of the pixels that comprise
             the moving object. 3) The rmse of the pixels that comprise the rest of the picture. 4) The rmse of the
             pixels not in the object if the move was random and not defined by the APNs final hidden layer.
    """
    images_where_there_is_no_object_in_Y = {'translate': {0: [62, 94], 1: [12, 23, 31, 59, 72], 2: []},
                                            'rotate': {0: [], 1: [], 2: []}}
    random_copied_from_pixels_indices = np.random.randint(low=0, high=31,
                                                                  size=copied_from_pixel_indices_all_images.shape)
    object_pixels_errors = []
    other_pixels_errors = []
    sum_of_errors = []
    image_indices = []

    num_of_samples = copied_from_pixel_indices_all_images.shape[0]

    for image_index in range(num_of_samples):
        object_colour = np.max(X[image_index])
        distance_to_correct_copy = 0

        to_add = False

        if data_type == 'translate' and image_index not in images_where_there_is_no_object_in_Y[data_type][distance]:
            to_add = True
            bounding_box_before_move, bounding_box_after_move, full_bounding_box = (
                get_bounding_boxes_for_X_Y(X, Y, object_colour, image_index))
            z = Z[image_index]
            object_pixels_translation = [0, 6] if z == 1 else [6, 0]

            pixels_after_move = np.array([(a[1], a[2]) for a in np.argwhere(Y == object_colour) if a[0] == image_index])
            for to_pixel in pixels_after_move:
                from_pixel = np.array([copied_from_pixel_indices_all_images[image_index, to_pixel[0], to_pixel[1], 0],
                                       copied_from_pixel_indices_all_images[image_index, to_pixel[0], to_pixel[1], 1]])
                dist_moved = np.abs(from_pixel - to_pixel)
                distance_to_correct_copy += np.sqrt(np.sum(np.power(dist_moved - object_pixels_translation, 2)))

        elif data_type == 'rotate':
            to_add = True
            bounding_box_before_move, bounding_box_after_move, full_bounding_box = (
                get_bounding_boxes_for_X_Y(X, Y, object_colour, image_index))
            for to_x in range(bounding_box_after_move[0, 1], bounding_box_after_move[1, 1] + 1, 1):
                for to_y in range(bounding_box_after_move[0, 0], bounding_box_after_move[1, 0] + 1, 1):
                    if Y[image_index, to_y, to_x] == object_colour:
                        from_pixel = np.array([copied_from_pixel_indices_all_images[image_index, to_y, to_x, 0],
                                                      copied_from_pixel_indices_all_images[image_index, to_y, to_x, 1]],
                                              dtype=int)
                        if X[image_index, from_pixel[0], from_pixel[1]] == object_colour:
                            pass
                        else:
                            to_pixel = np.array([to_y, to_x])
                            distance_to_correct_copy += np.sqrt(np.sum(np.power(from_pixel - to_pixel, 2)))

        if to_add:
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

    return (np.array(image_indices), np.array(object_pixels_errors),
            np.array(other_pixels_errors), random_error)

def get_errors_for_all_models_types_and_distances():

    all_image_indices = []
    all_object_pixels_errors = []
    all_other_pixels_errors = []
    all_random_errors = []
    all_labels = []
    for model_index in [0, 1]:
        for data_type in ['translate', 'rotate']:
            for distance in [0, 1, 2]:
                print(f'Model: {model_index}, Type: {data_type}, Distance: {distance}')
                X, Z, Y, copied_from_pixel_indices_all_images = load_data(model_index, data_type, distance)
                image_indices, object_pixels_errors, other_pixels_errors, random_error = \
                    get_non_compositional_errors(data_type, distance, X, Y, Z, copied_from_pixel_indices_all_images)

                all_labels.append(f'{"Axial Pointer Linear" if model_index==0 else "Axial Pointer"},\n{data_type},\n'
                                  f'D:{distance}')
                all_image_indices.append(image_indices)
                all_object_pixels_errors.append(object_pixels_errors)
                all_other_pixels_errors.append(other_pixels_errors)
                all_random_errors.append(random_error)

    return all_labels, all_image_indices, all_object_pixels_errors, all_other_pixels_errors, all_random_errors




all_labels, all_image_indices, all_object_pixels_errors, all_other_pixels_errors, all_random_errors = \
    get_errors_for_all_models_types_and_distances()


mean_object_error = []
std_object_error = []
mean_non_object_error = []
std_non_object_error = []

for i in range(len(all_labels)):
    mean_object_error.append(np.mean(all_object_pixels_errors[i]) / np.mean(all_random_errors[i]))
    mean_non_object_error.append(np.mean(all_other_pixels_errors[i]) / np.mean(all_random_errors[i]))
    std_object_error.append(np.std(all_object_pixels_errors[i]))
    std_non_object_error.append(np.std(all_other_pixels_errors[i]))


fig, (ax_o, ax_no) = plt.subplots(nrows=2, sharex=True)
bottom = 0
p1 = ax_o.bar(all_labels, height=mean_object_error, bottom=bottom)
p2 = ax_no.bar(all_labels, height=mean_non_object_error, bottom=bottom)
ax_no.tick_params(axis='x', labelrotation=90, labelsize=35)
ax_no.set_xlabel('Model, Type, Distance', {'size': 60})
ax_o.set_ylim(0, 0.5)
ax_no.set_ylim(0, 0.5)
ax_no.tick_params(axis='y', labelsize=40)
ax_o.tick_params(axis='y', labelsize=40)


from matplotlib.patches import FancyArrowPatch, ArrowStyle

fig, ax = plt.subplots()
arrow_style = ArrowStyle.Simple(head_length=.8, head_width=.8, tail_width=.2)
for o, no in zip(mean_object_error, mean_non_object_error):
    arrow = FancyArrowPatch((0, 0), (no, o), arrowstyle=arrow_style, mutation_scale=20)
    ax.add_patch(arrow)
ax.set_xlim(-0.1, 1)
ax.set_ylim(-0.1, 1)



model_index = 0
data_type = 'rotate'
distance = 1
X, Z, Y, copied_from_pixel_indices_all_images = load_data(model_index, data_type, distance)
image_indices,  object_pixels_errors, other_pixels_errors, random_error = get_non_compositional_errors(data_type, distance, X, Y, Z, copied_from_pixel_indices_all_images)
plt.plot(image_indices, object_pixels_errors, image_indices, other_pixels_errors, image_indices, random_error[image_indices])
plt.legend(('object', 'other', 'random'))
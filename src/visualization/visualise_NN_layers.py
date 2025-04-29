

from typing import List, Tuple
import matplotlib.pyplot as plt

from experiments.data.generation.data_to_tasks import SavedExperimentalDataToTask
from visualization.basic_visualisation_of_data import plot_data
import numpy as np


def visualise_input_data(data_filepath: str, task_index: int, show: str = 'in and out'):
    """
    Visualise the task, either all of it (as a Task) or just the input - output images the network works with
    :param data_filepath: The file to the data
    :param task_index: The index of the batch to visualise
    :param show: 'in and out' means show only the input and output images, 'task' means show the full task, 'everything' means show both the task and the input and output images
    :return:
    """
    tasks = SavedExperimentalDataToTask(data_filepath)

    if show in ['everything', 'task only']:
        tasks.show(task_index)

    if show in ['everything', 'in and out']:
        tasks.get_task(task_index).test_input_canvas.show()
        tasks.get_task(task_index).test_output_canvas.show()


def visualise_attention_results_with_colours(position_colours_in_all_images: np.ndarray | List,
                                             position_colours_out_all_images: np.ndarray | List,
                                             im_index: int) -> Tuple[plt.Figure, plt.axis, plt.axis]:
    """
    A visualisation of the results of the pixel_to_pixel_from_hw_attention_logits_matrix_based function.
    :param position_colours_in_all_images: The array where each element has a value that is a colour corresponding to the position of the element on the input image.
    :param position_colours_out_all_images: The array where each element has a value that is the colour of the pixel in the input image that got copied in the position of the output image with the same coordinates as the emelemnt of this array.
    :param im_index: Given that the position_colours_in_all_images and position_colours_out_all_images are batches of images, which input - output pair to visualise.
    :return: The generated figure and the two axis.
    """
    position_colours_in = position_colours_in_all_images[im_index]
    position_colours_out = position_colours_out_all_images[im_index]

    width = position_colours_in.shape[1]
    height = position_colours_in.shape[0]

    xmin = - 0.5
    xmax = width - 0.5
    ymin = - 0.5
    ymax = height - 0.5
    extent = (xmin, xmax, ymin, ymax)

    f = plt.figure()
    a1 = f.add_subplot(1, 2, 1)
    a2 = f.add_subplot(1, 2, 2)
    a1.imshow(position_colours_in, origin='lower', extent=extent, interpolation='None', aspect='equal')
    a2.imshow(position_colours_out, origin='lower', extent=extent, interpolation='None', aspect='equal')
    a1.vlines(np.arange(extent[0], extent[1]), ymin=extent[2], ymax=extent[3], colors='#BFBFBF', linewidths=0.3)
    a1.hlines(np.arange(extent[2], extent[3]), xmin=extent[0], xmax=extent[1], colors='#BFBFBF', linewidths=0.3)
    a2.vlines(np.arange(extent[0], extent[1]), ymin=extent[2], ymax=extent[3], colors='#BFBFBF', linewidths=0.3)
    a2.hlines(np.arange(extent[2], extent[3]), xmin=extent[0], xmax=extent[1], colors='#BFBFBF', linewidths=0.3)

    return f, a1, a2


def visualise_attention_results_with_index_annotations(copied_from_pixel_indices_all_images: List[np.ndarray],
                                                       im_index: int):
    copied_from_pixel_indices = copied_from_pixel_indices_all_images[im_index]

    x = np.array([[i] * 32 for i in range(32)]).flatten()
    y = np.array(list(range(32)) * 32).flatten()

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=0.5)

    for i, j in zip(x, y):
        txt = str(copied_from_pixel_indices[i, j, :].astype(int))
        ax.annotate(txt, (i, j), fontsize=10)


def visualise_array_as_canvas(array: np.ndarray) -> plt.Figure:
    array = np.round(array)

    width = array.shape[1]
    height = array.shape[0]

    xmin = - 0.5
    xmax = width - 0.5
    ymin = - 0.5
    ymax = height - 0.5
    extent = (xmin, xmax, ymin, ymax)

    f = plot_data(array.astype(int), extent)

    return f


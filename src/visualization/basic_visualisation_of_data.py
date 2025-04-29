
import numpy as np
import matplotlib.pyplot as plt
from structure.geometry.basic_geometry import Colour


def data_to_colour(pixels):
    pixels = np.array(pixels, np.uint8)
    size = pixels.shape
    result = np.zeros((size[0], size[1], 4)).astype(float)
    for i in range(size[0]):
        for j in range(size[1]):
            result[i, j, :] = np.array(Colour.map_int_to_colour[pixels[i, j]], float)
    return result


def add_vhlines_to_plot(axis, data, extent, thin_lines: bool = False):
    data = np.array(data)
    num_y = data.shape[0]
    num_x = data.shape[1]
    if not thin_lines:
        linewidths = 0.5 if np.max([num_y, num_x]) > 6 else 1.5
    else:
        linewidths = 0.1
    axis.vlines(np.arange(extent[0], extent[1]), ymin=extent[2], ymax=extent[3], colors='#BFBFBF', linewidths=linewidths)
    axis.hlines(np.arange(extent[2], extent[3]), xmin=extent[0], xmax=extent[1], colors='#BFBFBF', linewidths=linewidths)

    return axis


def plot_data(pixels, extent, axis: plt.Axes | None = None, thin_lines: bool = False):
    if axis is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        ax = axis
    pixels_for_visualisation = data_to_colour(pixels)
    ax.imshow(pixels_for_visualisation, origin='lower', extent=extent, interpolation='None', aspect='equal')
    if thin_lines:
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.1)
        ax.tick_params(axis='both', labelsize=2, grid_linewidth=0.1)
    ax = add_vhlines_to_plot(ax, pixels, extent, thin_lines)

    if axis is not None:
        return ax

    return fig, ax


def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=( 4 *n ,8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        t_in = data_to_colour(t_in)
        t_out = data_to_colour(t_out)
        axs[0][fig_num].imshow(t_in)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        t_in = np.array(t_in)
        #add_vhlines_to_plot(axs[0][fig_num], t_in)

        axs[1][fig_num].imshow(t_out)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        t_out = np.array(t_out)
        #add_vhlines_to_plot(axs[1][fig_num], t_out)

        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in = np.array(t["input"])
        t_in = data_to_colour(t_in)
        axs[0][fig_num].imshow(t_in)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        t_in = np.array(t_in)
        #add_vhlines_to_plot(axs[0][fig_num], t_in)

        fig_num += 1

    plt.tight_layout()


from __future__ import annotations

from typing import List, Tuple, Callable

import numpy as np

from matplotlib import pyplot as plt

from structure.canvas.canvas import Canvas
from structure.task.task import Task


class ARCLogicTask(Task):
    def __init__(self, objects_placement_function: Callable,
                 logic_function: Callable,
                 canvas_sizes_or_grid_stats: List[Dimension2D] | List[Tuple[int, int, int, int]],
                 num_of_outputs: int = 10, debug: bool = False):
        """
        A random generator of a full task. It requires the way to place objects in the input canvas of every example and
        the logic to run to produce the output canvas of each example. It will use those two functions to try and
        generate num_of_outputs number of examples. If either of these functions produce a None (failed to either place
        objects appropriately or to solve the logic) then that example will be thrown away. The generator will keep
        trying until it has the required number of examples or has made 10000 unsuccessful efforts.
        :param objects_placement_function: The function that will place objects in the input canvases. It should have both input and output the input canvas. If the output is None then that example will be ignored.
        :param logic_function: The function that should take as input the input canvas and solve the logic producing the output canvas. If the output is None then that example will be ignored.
        :param canvas_sizes_or_grid_stats: A list (as big as the number of examples in the task) of either the size of the input canvases (List[Dimension2D]) or the Tuple that turns the canvas into a grid (List[Tuple[int, int, int, int]]). The four ints are: the number of tiles along the x, the number of tiles along the y, the size of a tile and the colour of the grid lines.
        :param num_of_outputs: The number of examples in the task
        :param debug: If True print out some stuff. Default = False
        """
        super().__init__(prob_of_background_object=0, number_of_io_pairs=num_of_outputs, run_generate_canvasses=False)

        self.debug = debug
        self.num_of_outputs = num_of_outputs

        self.output_canvases = []
        self.input_canvases = []

        correct = 0
        wrong = 0

        while correct < num_of_outputs and wrong < 10000:
            if isinstance(canvas_sizes_or_grid_stats[0], Dimension2D):
                size = canvas_sizes_or_grid_stats[correct]
                input_canvas = Canvas(size=size, _id=correct * 2)
            elif isinstance(canvas_sizes_or_grid_stats[0], Tuple):
                grid_stats: Tuple[int, int, int, int] = canvas_sizes_or_grid_stats[correct]
                input_canvas = Canvas(as_grid_x_y_tilesize_colour=grid_stats)

            placement_func_return = objects_placement_function(input_canvas)
            if isinstance(placement_func_return, Tuple):
                input_canvas, input_description = placement_func_return
                self.input_descriptions.append(input_description)
            else:
                input_canvas = placement_func_return

            while len(input_canvas.objects) == 0:
                placement_func_return = objects_placement_function(input_canvas)
                if isinstance(placement_func_return, Tuple):
                    input_canvas, input_description = placement_func_return
                    self.input_descriptions.append(input_description)
                else:
                    input_canvas = placement_func_return

            if input_canvas is not None:
                logic_func_return = logic_function(input_canvas)

                if isinstance(logic_func_return, Tuple):
                    output_canvas, output_description = logic_func_return
                    self.output_descriptions.append(output_description)
                else:
                    output_canvas = logic_func_return

                if output_canvas is not None:
                    self.input_canvases.append(input_canvas)
                    self.output_canvases.append(output_canvas)
                    correct += 1
                else:
                    wrong += 1
            else:
                wrong += 1

    def create_20x32x32_data_arrays(self) -> np.ndarray:
        """
        Create a 20 x 32 x 32 array from the task's input and output canvasses. It assumes that the task has a maximum
        of 10 input/output examples. The order is array[0] = task.input_canvasses[0], array[1] = task.output_canvasses[0],
        array[2] = task.input_canvasses[1] etc. The last example always goes at the end of the array:
        array[18] = task.input_canvasses[-1], array[19] = task.output_canvasses[-1].
        If there are fewer than 10 examples then the inbetween array slices remain 0.
        :return: The np.ndarray
        """
        assert len(self.input_canvases) <= 20, print()
        input = np.zeros((20, 32, 32))
        output = np.zeros((20, 32, 32))

        for i in range(len(self.input_canvases)):
            input[i, :, :] = self.input_canvases[i].full_canvas
            output[i, :, :] = self.output_canvases[i].full_canvas

        result = np.zeros((20, 32, 32))
        for i in range(0, 2 * len(self.input_canvases) - 2, 2):
            result[i, :, :] = input[i // 2, :, :]
            result[i + 1, :, :] = output[i // 2, :, :]

        result[-2, :, :] = input[len(self.input_canvases)-1, :, :]
        result[-1, :, :] = output[len(self.input_canvases)-1, :, :]

        return result

    def show(self, canvas_index: int | str = 'all', save_as: str | None = None, two_cols: bool = False):
        thin_lines = True
        if save_as is None:
            thin_lines = False

        fig = plt.figure(figsize=(6, 16))
        index = 1
        ncoloumns = 4

        nrows = int(np.ceil((2 * self.num_of_outputs + 1) / ncoloumns))
        for i, o in zip(self.input_canvases, self.output_canvases):
            i.show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index, thin_lines=thin_lines)
            o.show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 1, thin_lines=thin_lines)

            index += 2



import numpy as np
from structure.canvas.canvas import Canvas
from structure.task.task import Task


class SavedExperimentalDataToTask:
    def __init__(self, path_to_npz_array: str):
        data = np.load(path_to_npz_array, allow_pickle=True)
        self.images_of_all_tasks = data['samples']
        self.descriptions_of_all_tasks = data['languages']

    def get_task(self, index: int) -> Task:
        task_images = self.images_of_all_tasks[index]
        task_description = self.descriptions_of_all_tasks[index]

        number_of_io_pairs = 9
        for i in range(20):
            if np.sum(task_images[:, :, i]) == 0:
                number_of_io_pairs = i // 2
                break

        task = Task(prob_of_background_object=0, number_of_io_pairs=number_of_io_pairs,
                    run_generate_canvasses=False)

        for i in range(2 * number_of_io_pairs):
            if i % 2 == 0:
                task.input_canvases.append(Canvas(actual_pixels=task_images[:, :, i]))
            else:
                task.output_canvases.append(Canvas(actual_pixels=task_images[:, :, i]))

        task.test_input_canvas = Canvas(actual_pixels=task_images[:, :, 18])
        task.test_output_canvas = Canvas(actual_pixels=task_images[:, :, 19])
        task.task_description = task_description
        task.id = task_description['id']

        return task

    def show(self, task_index: int, save_as: str | None = None):
        self.get_task(task_index).show(save_as=save_as)

    def task_description(self, task_index:int) -> str:
        return self.get_task(task_index).task_description
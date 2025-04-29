
import os
from pathlib import Path

import click
import numpy as np
from os.path import join

from matplotlib import pyplot as plt
from tqdm import tqdm

from experiments.analysis.saved_data_sets_to_tasks import SavedExperimentalDataToTask


@click.command()
@click.argument('data_file', type=click.Path())
@click.argument('save_to_path', type=click.Path())
@click.argument('num_of_images_to_save', type=click.INT)
def main(data_file: str, save_to_path: str, num_of_images_to_save: int):
    world_model_type = 'translate' if 'translate' in data_file else 'rotate'
    data_set_type = 'train' if 'train' in data_file else 'test'
    distance = '_' if data_set_type == 'train' else int(data_file[data_file.find('.npz')-1])

    # Generate a set of Tasks (like ARC Tasks) from the data set
    all_tasks = SavedExperimentalDataToTask(data_file)

    indices = np.random.choice(np.arange(all_tasks.images_of_all_tasks.shape[0]), size=num_of_images_to_save)

    print(f'Creating {num_of_images_to_save} images')

    if not os.path.exists(save_to_path):
        Path(save_to_path).mkdir(parents=True, exist_ok=True)
    for i in tqdm(indices):
        filename = join(save_to_path, f'{world_model_type}_{data_set_type}{distance}_image_{i}.png')
        all_tasks.show(task_index=i, save_as=filename)


if __name__ == '__main__':
    main()
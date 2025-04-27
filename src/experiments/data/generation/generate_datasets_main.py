
import json
from typing import Callable, Dict, Tuple, List

import click
import logging

from dotenv import find_dotenv, load_dotenv

import numpy as np
import concurrent.futures
from tqdm import tqdm
import os
import time

# Import all the possible generating functions that the main can call
from src.experiments.data.generation.compositional_datasets import generate_compositional_datasets

NUM_OF_TASKS_IN_SAMPLE = 100


def generate_20x32x32_samples_dataset(seed, gen_function: Callable, gen_functions_args: Dict) -> Tuple[np.ndarray, List]:
    """
    The function that calls the Task generating function NUM_OF_TASKS_IN_SAMPLE times (each of which returns a Task) and
    repackages the Tasks' info into two arrays, the images and the descriptions. The images array is a [NUM_OF_TASKS_IN_SAMPLE, 32, 32, 20] int8 array.
    The examples of each Task are given in the 20 panes as input[0], output[0],... input[9], output[9]. If the Task has
    fewer than 10 input, output pairs then the last two panes (19th and 20th) are the last pair of the Task and there
    will be a series of empty panes (all zeros) between the initial filled ones and those last two.
    :param seed: A number to set the random seed
    :param gen_function: The Task generating function which will create a Task
    :param gen_functions_args: The arguments of the Task generating function minus the num_of_examples. The num_of_examples if randomly set in this function.
    :return: A tuple of the two arrays.
    """
    np.random.seed(os.getpid() + int(time.time()) + seed)

    task_images = []
    tasks_descriptions = []

    for task_name in range(NUM_OF_TASKS_IN_SAMPLE):
        success = False
        while not success:
            try:
                num_of_examples = np.random.randint(3, 11)  # Randomly choose n between 3 and 10
                task = gen_function(num_of_examples, **gen_functions_args)
                success = True
            except:
                pass
        images_array = task.create_20x32x32_data_arrays()
        task_description = task.task_description
        task_images.append(images_array)
        tasks_descriptions.append(task_description)

    task_images = np.array(task_images, dtype="int8")

    return task_images, tasks_descriptions


@click.command()
@click.argument('num_samples', type=click.INT)
@click.argument('output_filepath', type=click.Path())
@click.argument('generating_function', type=click.STRING)
@click.argument('generating_functions_args', type=click.UNPROCESSED)
def main(generating_function, generating_functions_args, output_filepath, num_samples):
    """
    The function to call from the command line in order to save data produced by a certain Task generating function.
    :param generating_function: The name of the Task generating function to be used. The function needs to have been
                                imported for this to work. For the paper the only function is generate_compositional_datasets
    :param generating_functions_args: The string of the dictionary (in json format) of the arguments of the Task
                                      generating function without the num_of_examples argument. E.g. for the
                                      generate_compositional_datasets gen function a possible
                                      generating_functions_args would be '{"distance": 0, "symmetric_objects": 1,
                                      "transformation_type": "translate", "second_object": 0}'.
    :param output_filepath: The full path where the generated data (arrays) will be stored. The resulting file is an .npz.
    :param num_samples: The number of samples of NUM_OF_TASKS_IN_SAMPLE Tasks. The final image array will be a
                        [num_samples * NUM_OF_TASKS_IN_SAMPLE, 32, 32, 20] int8 array
    :return: Nothing.
    """
    images_array = []
    languages_array = []
    num_workers = 50

    generating_functions_args = json.loads(generating_functions_args)
    generating_function = globals()[generating_function]

    kargs = {'gen_function': generating_function, 'gen_functions_args': generating_functions_args}
    # Use ProcessPoolExecutor for true multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Creating a list of futures with the seed passed to each task
        futures = [executor.submit(generate_20x32x32_samples_dataset, i, **kargs) for i in range(num_samples)]

        # Using tqdm to show progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_samples):
            try:
                images, language = future.result()
                images_array.append(images)
                languages_array.extend(language)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # Convert results list into a numpy array and save it
    images_array = np.concatenate(images_array)
    images_array = np.transpose(images_array, (0, 2, 3, 1))

    # Save results to .npz file
    np.savez(output_filepath, samples=images_array, languages=languages_array)
    print(f"Saved {len(images_array)} samples to {output_filepath}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
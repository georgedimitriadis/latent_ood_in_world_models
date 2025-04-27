import logging
import os
from datetime import datetime
from pathlib import Path


import click
import keras
import numpy as np
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from tqdm.keras import TqdmCallback

from src.models.nn_models import axial_pointer_network
from src.models.lm import b_acc_s
from src.models.utils import make_data_from_train_sparse
from src.models.utils import write_dict_to_csv


def transform_array(language):
    merged = [[l["id"]] for l in language]
    input_array = np.array(merged, dtype="int8")
    return input_array


@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('train_data_filepath', default='data/processed/task_comp', type=click.Path())
@click.argument('test_data_filepath', default='data/processed/task_comp', type=click.Path())
@click.argument('logs_filepath', default='./logs', type=click.Path())
@click.argument('use_tensorboard', default=False, type=click.BOOL)
def main(output_filepath, train_data_filepath, test_data_filepath, logs_filepath, use_tensorboard):
    """
    Call the training of the compositional model
    :param output_filepath: The path where the model will be saved as output_filepath/model_name.keras
    :param train_data_filepath: The file path of the train data set. The full path is train_data_filepath/train.npz since it assumes the name of the train data file is train.npz
    :param test_data_filepath: The file path of the test data sets. The full paths are test_data_filepath/test_d{i}.npz since it assumes that the names of the test sets are test_d0.npz, test_d1.npz and test_d2.npz
    :param logs_filepath: The path where the log file will be saved as logs_filepath/composition_log_model_name.csv
    :param use_tensorboard: If True then stuff get saved in a tensorboard format
    :return:
    """

    model_name = "axial_pointer_network"
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    model = axial_pointer_network(
                                  n_tasks=4,
                                  )
    model.summary()

    log_filename = f"{logs_filepath}/task_comp_log_{date_time}_{model_name}.csv"


    optimizer = keras.optimizers.AdamW(
        learning_rate=0.0001,
        weight_decay=0.004)

    model.compile(optimizer=optimizer, loss="SparseCategoricalCrossentropy",
                  metrics=[b_acc_s, "accuracy"])


    print("Loading train data...")
    data_train = np.load(f'{train_data_filepath}/train.npz', allow_pickle=True)
    images_array = data_train["samples"]
    language = data_train["languages"]

    X_train, _, y_train = make_data_from_train_sparse(images_array)
    Z_train = transform_array(language)

    print("Loading test data...")
    test = [0,0,0]
    for i in range(len(test)):
        test_data = np.load(f"{test_data_filepath}/test_d{i}.npz",
                            allow_pickle=True)
        images_array = test_data["samples"]
        language = test_data["languages"]
        X, _, y = make_data_from_train_sparse(images_array)
        Z = transform_array(language)
        test[i] = X, Z, y

    for i in range(0, 200):
        model.fit(x=[X_train, Z_train], y=y_train,
                  epochs=1,
                  verbose=0, batch_size=200,
                  callbacks=[TqdmCallback()])

        for distance in tqdm([0,1,2]):
            X, Z, Y = test[distance]

            score = model.evaluate(x=[X, Z], y=Y,
                                   return_dict="true", verbose=False)

            if ((i + 1) % 10) == 0:
                model.save(f"{output_filepath}/{model_name}.keras")

            score["epoch"] = i
            score["distance"] = distance
            print(i, distance, score)
            write_dict_to_csv(score, log_filename)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

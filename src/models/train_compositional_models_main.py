
import logging
from datetime import datetime
from os.path import join
from pathlib import Path


import click
import keras
import numpy as np
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from tqdm.keras import TqdmCallback

from models.nn_models import (axial_pointer_network, pure_cnn, transformer, mlp_nn)
from models.lm import b_acc_s
from models.utils import make_data_from_train_sparse, plot_images_with_action
from models.utils import write_dict_to_csv


def visualise_model(model: keras.Model, to_file: str = "./model.png"):
    keras.utils.plot_model(model, to_file=to_file, show_shapes=True, show_dtype=False, show_layer_names=True,
                           rankdir="TB", expand_nested=False, dpi=200, show_layer_activations=False,
                           show_trainable=False)


def visualise_images_with_action(model, X, Z, Y, logs_filepath, distance):
    Y_prime = model.predict(x=[X, Z])
    for j, (x, a, y, y_prime) in enumerate(list(zip(X, Z, Y, Y_prime))[:100]):
        y_prime = np.argmax(y_prime, axis=-1)
        filename_id = f"{logs_filepath}/figures/distance_{distance}_{j}_{model.name}"
        plot_images_with_action(x, a, y, y_prime, filename_id)


@click.command()
@click.argument('model_type', default='axial_point_network_full', type=click.STRING)
@click.argument('num_epochs', default=100, type=click.INT)
@click.argument('with_language', default=False, type=click.BOOL)
@click.argument('with_mask', default=False, type=click.BOOL)
@click.argument('action_bits_indices', default='1', type=click.STRING)
@click.argument('save_model_filepath', type=click.Path())
@click.argument('train_data_filepath', default='data/processed/compositional', type=click.Path())
@click.argument('test_data_filepath', default='data/processed/compositional', type=click.Path())
@click.argument('logs_filepath', default='./logs', type=click.Path())
def main(model_type, with_language, with_mask, num_epochs, action_bits_indices,
         save_model_filepath, train_data_filepath, test_data_filepath, logs_filepath):
    """
    Call the training of the compositional model
    :param model_type: The model type string. Can be axial_point_network_lines, axial_point_network_full, pure_cnn, transformer, mlp_nn
    :param with_language: If True then use the language mechanism in the NN to generate the task index. If False no such mechanism is used.
    :param with_mask: If True then the axial point networks will create a final mask to copy stuff from a new image.
    :param num_epochs: Number of epochs
    :param action_bits_indices: The indices of the language['bits'] array values (bits) that should be used to define the action.
    :param save_model_filepath: The path where the model will be saved as output_filepath/model_type.keras
    :param train_data_filepath: The file path of the train data set. The full path is train_data_filepath/train.npz since it assumes the name of the train data file is train.npz
    :param test_data_filepath: The file path of the test data sets. The full paths are test_data_filepath/test_d{i}.npz since it assumes that the names of the test sets are test_d0.npz, test_d1.npz and test_d2.npz
    :param logs_filepath: The path where the log file will be saved as logs_filepath/composition_log_model_type.csv
    :return:
    """

    if action_bits_indices != '-1':
        action_bits_indices = [int(i) for i in action_bits_indices]
    else:
        action_bits_indices = [-1]

    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if 'axial_point_network' in model_type:
        if 'full' in model_type:
            line_features = False
            n_mlp_units = 32
        elif 'lines' in model_type:
            line_features = True
            n_mlp_units = 256
        model = axial_pointer_network(n_mlp_units=n_mlp_units, n_words_per_sentence=1,
                                      n_tasks=2 * len(action_bits_indices),
                                      with_language=with_language, with_mask=with_mask, line_features=line_features)
    elif model_type == 'pure_cnn':
        model = pure_cnn(base_filters=54, encoder_filters=128, n_tasks=2 * len(action_bits_indices))
    elif model_type == 'transformer':
        model = transformer(n_mlp_layers=4, with_language=with_language, n_tasks=2 * len(action_bits_indices))
    elif model_type == 'mlp_nn':
        model = mlp_nn(n_mlp_layers=4, projection_dim=32, n_tasks=2 * len(action_bits_indices),
                       with_language=with_language)
    model.summary()
    model.name = model_type

    log_filename = f"{logs_filepath}/composition_log_{date_time}_{model_type}.csv"

    def transform_language_bits_to_action_index(language, action_bits_indices):
        bits = [[l['bits'][i] for i in action_bits_indices] for l in language]
        merged = [[int(''.join(map(str, bit)), 2)] for bit in bits]

        #merged = [l["bits"][-1:] for l in language]
        input_array = np.array(merged, dtype="int8")
        return input_array

    optimizer = keras.optimizers.AdamW(
        learning_rate=0.0001,
        weight_decay=0.004)

    model.compile(optimizer=optimizer, loss="SparseCategoricalCrossentropy",
                  metrics=[b_acc_s, "accuracy"])

    visualise_model(model, join(logs_filepath, model_type + '.png'))

    print("Loading train data...")
    data_train = np.load(f'{train_data_filepath}/train.npz', allow_pickle=True)
    images_array = data_train["samples"]
    language = data_train["languages"]

    X_train, _, y_train = make_data_from_train_sparse(images_array)
    Z_train = transform_language_bits_to_action_index(language, action_bits_indices)

    print("Loading test data...")
    test = [0, 0, 0]
    for i in range(len(test)):
        test_data = np.load(f"{test_data_filepath}/test_d{i}.npz",
                            allow_pickle=True)
        images_array = test_data["samples"]
        language = test_data["languages"]
        X, _, y = make_data_from_train_sparse(images_array)
        Z = transform_language_bits_to_action_index(language, action_bits_indices)
        test[i] = X, Z, y

    for i in range(0, num_epochs):
        model.fit(x=[X_train, Z_train], y=y_train,
                  epochs=1,
                  verbose=0, batch_size=100,
                  callbacks=[TqdmCallback()])

        for distance in tqdm([0, 1, 2]):
            X, Z, Y = test[distance]
            score = model.evaluate(x=[X, Z], y=Y,
                                   return_dict="true", verbose=False)
            #

            if ((i + 1) % 20) == 0:
                model.save(f"{save_model_filepath}/{model_type}.keras")
                visualise_images_with_action(model, X, Z, Y, logs_filepath, distance)

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

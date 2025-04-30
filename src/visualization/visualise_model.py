
import logging
import warnings

# Suppress all logger messages
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib.image").setLevel(logging.CRITICAL)

# Suppress all warnings
warnings.filterwarnings("ignore")

import click

import logging

from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import keras
from matplotlib import pyplot as plt
from models.lm import b_acc_s


def visualise_model(model_filepath: str):
    keras.config.enable_unsafe_deserialization()

    logging.info("Loading saved_models")

    original_model = keras.models.load_model(f"{model_filepath}")

    # Print the layers to see what we're working with
    print("Available layers:")
    for i, layer in enumerate(original_model.layers):
        print(f"{i}: {layer.name}")

    selected_layers = [layer.name for layer in original_model.layers]
    # Get the output tensors for our selected layers
    output_tensors = [
        original_model.get_layer(layer.name).output
        for layer in original_model.layers
    ]

    # Create new model with multiple outputs
    new_model = keras.Model(
        inputs=original_model.input,
        outputs=output_tensors,
        name="multi_output_model"
    )

    # Print information about the outputs
    print("\nNew model outputs:")
    for i, output in enumerate(new_model.outputs):
        print(f"Output {i} ({selected_layers[i]}): Shape = {output.shape}")

    # If you want to freeze the weights
    for layer in new_model.layers:
        layer.trainable = False


@click.command()
@click.argument('model_filepath', type=click.File())
def main(model_filepath):
    model_filepath = model_filepath.name
    visualise_model(model_filepath=model_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

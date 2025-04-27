

import os
from typing import List, Tuple, Any

from tqdm import tqdm
import keras.api.ops as ops
from models.utils import make_data_from_train_sparse
import numpy as np
import keras
from models.lm import b_acc_s


def load_data(data_filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the np.ndarrays in the data_filepath file (an .npz file)
    :param data_filepath: The file storing the data
    :return: The images and the language descriptions used to train or test a network
    """
    def transform_array(language):
        merged = [l["bits"][-1:] for l in language]
        input_array = np.array(merged, dtype="int8")
        return input_array

    data_train = np.load(data_filepath, allow_pickle=True)
    images_array = data_train["samples"]
    language = data_train["languages"]

    input_images, _, output_images = make_data_from_train_sparse(images_array)
    input_language = transform_array(language)

    return input_images, input_language, output_images


def run_model(model_filepath: str, data_filepath: str) -> np.ndarray:
    """
    Run the model saved in the model_filepath on the data saved in the data_filepath and return the resulting images.
    :param model_filepath: The file of the model (a .keras file)
    :param data_filepath: The file of the input data (a .npz file)
    :return: The output images
    """
    os.environ['KERAS_BACKEND'] = 'jax'
    keras.config.enable_unsafe_deserialization()
    original_model = keras.models.load_model(f"{model_filepath}")

    X, Z, _ = load_data(data_filepath)

    y = original_model.predict([X, Z])
    y = un_one_hot(y)

    return y


def find_prediction_error_indices(model_filepath: str, data_filepath: str) -> np.ndarray:
    y_hat = run_model(model_filepath=model_filepath, data_filepath=data_filepath)
    _, _, y = load_data(data_filepath=data_filepath)

    return np.argwhere(y != y_hat)


def load_models_layers(model_filepath: str, data_filepath: str, batch_size: int = 1000,
                       selected_layers: List[str] = ('attention_logits', 'mask', 'new_image', 'smooth_blend')) -> Any:
    """
    Extract the activations of selected_layers of the model in model_filepath after it has run on the data_filepath data.
    :param model_filepath: The model
    :param data_filepath: The input to the data
    :param selected_layers: The names of the layers to get the activations of
    :return: A list of the activations of all requested layers
    """
    os.environ['KERAS_BACKEND'] ='jax'

    # Get the model's important layers' weights in a new model
    keras.config.enable_unsafe_deserialization()
    original_model = keras.models.load_model(f"{model_filepath}")

    output_tensors = [
        original_model.get_layer(layer_name).output
        for layer_name in selected_layers
    ]
    new_model = keras.Model(
        inputs=original_model.input,
        outputs=output_tensors,
        name="multi_output_model"
    )

    # Get the data to be used for prediction
    X, Z, _ = load_data(data_filepath)

    # Use the data to calculate what the model would spit out at the selected layers
    output_layers = new_model.predict([X, Z], batch_size=batch_size)

    return output_layers


def pixel_to_pixel_from_hw_attention_logits_rules_based(batch_size: int, height: int, width: int,
                                                        attention_logits: np.ndarray | List,
                                                        examples_from_batch: List[int] = 0) \
        -> List[np.ndarray]:
    """
    Generates a list of arrays (one for each image in the examples_from_batch). Each array is a height x width one that
    in each element carries the indices (height x width) of the pixel in the input image that was copied over to the
    h,w pixel of the output image. It does this by following the rules of how the attention weights in the
    attention_logits store the copying mechanism from input pixels to output pixels.
    :param batch_size: The size of the batch for the attention_logits.
    :param height: The height of the image.
    :param width: The width of the image.
    :param attention_logits: The attention_logits coming from the previous NN layers.
    :param examples_from_batch: Which images to calculate the result for (this is a slow algorithm).
    :return: The List of arrays carrying the info on how input pixels get copied to output pixels.
    """

    row_size = height * height
    col_size = width * width
    attention_weights_h = ops.reshape(
        attention_logits[:, :row_size],
        (batch_size, height, height)
    )
    attention_weights_w = ops.reshape(
        attention_logits[:, row_size:row_size + col_size],
        (batch_size, width, width)
    )

    attention_weights_h = ops.argmax(attention_weights_h, axis=-1)
    weights_h = ops.one_hot(attention_weights_h,
                            attention_weights_h.shape[-1])

    attention_weights_w = ops.argmax(attention_weights_w, axis=-1)
    weights_w = ops.one_hot(attention_weights_w,
                            attention_weights_w.shape[-1])

    copied_from_pixel_indices_all_images = []
    for b in tqdm(examples_from_batch):
        copied_from_pixel_indices = np.zeros((height, width, 2))
        for r_in in tqdm(range(height)):
            for c_in in range(width):
                for w_1st_index in range(height):
                    for h_1st_index in range(width):
                        if weights_w[b][w_1st_index, c_in] == 1 and weights_h[b][h_1st_index, r_in] == 1:
                            w = int((w_1st_index * width + h_1st_index) // height)
                            z = (w_1st_index * width + h_1st_index) % height
                            copied_from_pixel_indices[z, w, :] = (r_in, c_in)
        copied_from_pixel_indices_all_images.append(copied_from_pixel_indices)

    return copied_from_pixel_indices_all_images


def pixel_to_pixel_from_hw_attention_logits_matrix_based(input_images: np.ndarray, attention_logits: np.ndarray | List,
                                                         examples_from_batch: List[int] = 0,
                                                         amount_of_green_for_shape: float = 0.2) -> Tuple[List, List, List]:
    """
    Generates three list of arrays (one for each image in the examples_from_batch). The first two arrays are images.
    The first of those two color codes each entry according to the index of the pixel (black - blue for the column
    axis and black - red for the row axis). In the second image each pixel is color coded according to which pixel
    from the input image has been copied to the output image. The third array is a height x width one that
    in each element carries the indices (height x width) of the pixel in the input image that was copied over to the
    h,w pixel of the output image. It does this by following the rules of how the attention weights in the
    attention_logits store the copying mechanism from input pixels to output pixels.
    :param amount_of_green_for_shape: How much to add a tinge of green to the pixels that form the shape of the image
    :param input_images: All the input images in a batch.
    :param attention_logits: The attention logits to copy the pixels form the input images to the output ones.
    :param examples_from_batch: Which of the input images of the batch to calculate the results for.
    :return: See description.
    """

    batch_size = input_images.shape[0]
    height = input_images.shape[1]
    width = input_images.shape[2]

    num_channels = 11

    row_size = height * height
    col_size = width * width
    attention_weights_h_list = []
    attention_weights_w_list = []

    attention_weights_h = ops.reshape(
        attention_logits[:, :row_size],
        (batch_size, height, height)
    )
    attention_weights_w = ops.reshape(
        attention_logits[:, row_size:row_size + col_size],
        (batch_size, width, width)
    )

    attention_weights_h = ops.argmax(attention_weights_h, axis=-1)
    attention_weights_h = ops.one_hot(attention_weights_h,
                                      attention_weights_h.shape[-1])

    attention_weights_w = ops.argmax(attention_weights_w, axis=-1)
    attention_weights_w = ops.one_hot(attention_weights_w,
                                      attention_weights_w.shape[-1])

    visualisations_in = []
    visualisations_out = []
    copied_from_pixel_indices_all_images = []
    for image_index in tqdm(examples_from_batch):
        batch_size = height * width
        data_in = np.zeros((batch_size, height, width))
        b = 0
        for i in range(height):
            for j in range(width):
                data_in[b, i, j] = 1
                b += 1

        image_in_oh = np.array(ops.one_hot(data_in, 11))

        weights_h_one_pix = np.array([attention_weights_h[image_index] for _ in range(height * width)])
        weights_w_one_pix = np.array([attention_weights_w[image_index] for _ in range(height * width)])

        image_h = ops.reshape(image_in_oh, (batch_size, height, width * num_channels))
        transformed_h = ops.matmul(weights_h_one_pix, image_h)  # Apply attention along height
        transformed_h = ops.reshape(transformed_h, (batch_size, width, width, num_channels))

        image_w = ops.transpose(transformed_h, (0, 2, 1, 3))  # Swap height and width
        image_w = ops.reshape(image_w, (batch_size, width, width * num_channels))  # Merge height & channels
        transformed_w = ops.matmul(weights_w_one_pix, image_w)  # Apply attention along width
        transformed_w = ops.reshape(transformed_w, (batch_size, width, height, num_channels))
        transformed_w = ops.transpose(transformed_w, (0, 2, 1, 3))  # Swap height and width back

        transformed_w = np.array(transformed_w)
        image_out = un_one_hot(transformed_w)

        vis_in = np.zeros((height, width, 3))
        vis_out = np.zeros((height, width, 3))
        copied_from_pixel_indices = np.zeros((height, width, 2))
        pixel_pane_index = 0
        for i in range(height):
            for j in range(width):
                g = amount_of_green_for_shape if input_images[image_index][i, j] > 1 else 0
                colour = (i / 32, g, j / 32)
                vis_in[i, j, :] = colour
                indices = np.argwhere(image_out[pixel_pane_index])
                vis_out[indices[:, 0], indices[:, 1], :] = colour

                for ind in indices:
                    copied_from_pixel_indices[ind[0], ind[1], :] = (i, j)

                pixel_pane_index += 1

        visualisations_in.append(vis_in)
        visualisations_out.append(vis_out)
        copied_from_pixel_indices_all_images.append(copied_from_pixel_indices)

    return visualisations_in, visualisations_out, copied_from_pixel_indices_all_images


def pixel_to_pixel_from_full_attention_logits_matrix_based(input_images: np.ndarray, attention_logits: np.ndarray | List,
                                                           examples_from_batch: List[int] = 0,
                                                           amount_of_green_for_shape: float = 0.2) -> Tuple[List, List, List]:
    batch_size = ops.shape(input_images)[0]
    height, width = ops.shape(input_images)[1], ops.shape(input_images)[2]

    # Calculate sizes for reshaping
    h_logits_size = height * width * height  # For each target (h,w), we have 'height' source row positions

    # Split the flattened logits into horizontal and vertical components
    h_logits_flat = attention_logits[:, :h_logits_size]
    w_logits_flat = attention_logits[:, h_logits_size:]

    # Reshape to proper dimensions
    h_logits = ops.reshape(h_logits_flat, [batch_size, height, width, height])  # [b, h, w, h]
    w_logits = ops.reshape(w_logits_flat, [batch_size, height, width, width])  # [b, h, w, w]

    h_probs = ops.argmax(h_logits, axis=-1)
    h_probs = ops.one_hot(h_probs, h_probs.shape[-1])

    w_probs = ops.argmax(w_logits, axis=-1)
    w_probs = ops.one_hot(w_probs, w_probs.shape[-1])

    visualisations_in = []
    visualisations_out = []
    copied_from_pixel_indices_all_images = []
    for image_index in tqdm(examples_from_batch):
        batch_size = height * width
        data_in = np.zeros((batch_size, height, width))
        b = 0
        for i in range(height):
            for j in range(width):
                data_in[b, i, j] = 1
                b += 1

        image_in_oh = np.array(ops.one_hot(data_in, 11))
        h_probs_for_all_single_pixels = np.tile(h_probs[image_index], (width * height, 1, 1, 1))
        w_probs_for_all_single_pixels = np.tile(w_probs[image_index], (width * height, 1, 1, 1))
        h_gathered = ops.einsum('piwh,phwc->piwc', h_probs_for_all_single_pixels, image_in_oh)
        w_gathered = ops.einsum('pijw,piwc->pijc', w_probs_for_all_single_pixels, h_gathered)

        image_out = un_one_hot(w_gathered)

        vis_in = np.zeros((height, width, 3))
        vis_out = np.zeros((height, width, 3))
        copied_from_pixel_indices = np.zeros((height, width, 2))
        pixel_pane_index = 0
        for i in range(height):
            for j in range(width):
                g = amount_of_green_for_shape if input_images[image_index][i, j] > 1 else 0
                colour = (i / 32, g, j / 32)
                vis_in[i, j, :] = colour
                indices = np.argwhere(image_out[pixel_pane_index])
                vis_out[indices[:, 0], indices[:, 1], :] = colour

                for ind in indices:
                    copied_from_pixel_indices[ind[0], ind[1], :] = (i, j)

                pixel_pane_index += 1

        visualisations_in.append(vis_in)
        visualisations_out.append(vis_out)
        copied_from_pixel_indices_all_images.append(copied_from_pixel_indices)

    return visualisations_in, visualisations_out, copied_from_pixel_indices_all_images


def un_one_hot(array: np.ndarray) -> np.ndarray:
    """
    The inverse of the one_hot functions
    :param array: The one_hot encoded array
    :return: An array that is encoded in R
    """
    batch_size = array.shape[0]
    height = array.shape[1]
    width = array.shape[2]
    array = np.sum(array * np.array([np.ones((batch_size, height, width)) * i for i in range(11)]).transpose((1, 2, 3, 0)),
                   axis=3)
    array = np.round(array)
    return array

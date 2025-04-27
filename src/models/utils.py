import warnings
import numpy as np
import keras
import sklearn.metrics
from typing import Tuple
activation = "relu"
from sklearn.linear_model import Ridge


def unpack_data(result_from_create_data_from_task: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inputs = []
    outputs = []
    for i, array in enumerate(result_from_create_data_from_task):
        if np.sum(array) > 0:
            if i % 2 == 0:
                inputs.append(array)
            else:
                outputs.append(array)

    return np.array(inputs), np.array(outputs)


def predict_train_lr(dense, batch_element, encoder_flat, model_full, train):
    batch_element = np.transpose(batch_element, (2, 0, 1))

    inputs, outputs = unpack_data(batch_element)

    inputs_oh = np.eye(11)[inputs]
    outputs_oh = np.eye(11)[outputs]

    lx_train = inputs_oh[:-1]
    ly_train = outputs_oh[:-1]

    lx_test = inputs_oh[-1:]
    ly_test = outputs_oh[-1:]

    inputs_encoded = encoder_flat(lx_train)
    outputs_encoded = encoder_flat(ly_train)
    clf = Ridge(fit_intercept=False)

    clf.fit(inputs_encoded, outputs_encoded)

    score = clf.score(inputs_encoded, outputs_encoded)

    dense.set_weights([clf.coef_])

    if (train):
        losses = model_full.train_on_batch(lx_test, ly_test, return_dict=True)
        return losses, score
    else:
        return model_full.predict(lx_test), score


def make_data_from_train(arc_data):
    b = (arc_data)
    y = np.eye(11, dtype="int8")[b[:, :, :, -1]]

    X = np.eye(11, dtype="int8")[b[:, :, :, -2:-1]]

    Z = np.eye(11, dtype="int8")[b[:, :, :, :-2]]

    my_X = X.reshape(X.shape[0], X.shape[1], X.shape[2],
                     X.shape[3] * X.shape[4])

    my_Z = Z.reshape(Z.shape[0], Z.shape[1], Z.shape[2],
                     Z.shape[3] * Z.shape[4])

    return my_X, my_Z, y


def masked_copy_fill(arr):
    new_arr = []
    for e in (arr):
        n = len(e)

        # Create a range array for the full length
        idx = np.arange(n)

        mask = []
        for k in e:
            if(k.sum() > 0.1):
                mask.append(True)
            else:
                mask.append(False)
        #print(mask)
        mask = np.array(mask)



        # Count number of True values without using mask as index
        n_true = np.sum(mask)

        # Create modulo indices for the entire array
        cycle_indices = idx % n_true
        #print(n, n_true, cycle_indices)

        result = e[cycle_indices]
        new_arr.append(result)


    return np.array(new_arr,dtype="int8")
def make_data_from_train_sparse(arc_data):
    b = (arc_data)
    y = b[:, :, :, -1]

    X = b[:, :, :, -2]

    Z = b[:, :, :, :-2]

    my_X = X

    my_Z = Z.transpose([0,3,1,2])

    # print(Z.shape,X.shape, y.shape)
    # exit()

    return my_X, my_Z, y


def extract_X_y(b, i):
    """
    Extract X and y from a NumPy array with specific indexing and stopping condition.

    Parameters:
    b (numpy.ndarray): Input 4D array

    Returns:
    tuple: (X, y) where X and y are lists of extracted slices
    """

    b = b[:, :, :, :-2]
    X = []
    y = []

    x_indices = list(range(0, b.shape[3], 2))
    y_indices = list(range(1, b.shape[3], 2))

    #print(x_indices, y_indices)



    for x_i, y_i in zip(x_indices, y_indices):
        bx_i = b[i, :, :, x_i]
        by_i = b[i, :, :, y_i]
        #print(by_i.shape, np.sum(by_i))
        if(np.sum(bx_i) == np.sum(by_i) == 0 ):
            break
        else:
            X.append(bx_i)
            y.append(by_i)


    X = np.array(X)
    y = np.array(y)

    #print(X.shape,y.shape)

    return X, y

def leave_one_out_cv(X, y):
    """
    Performs leave-one-out cross-validation by generating all possible train/test splits.

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, ...)
    y : numpy.ndarray
        Target values of shape (n_samples, ...)

    Returns:
    --------
    X_train : numpy.ndarray
        Training features for each fold with shape (n_samples, n_samples-1, ...)
    y_train : numpy.ndarray
        Training targets for each fold with shape (n_samples, n_samples-1, ...)
    X_test : numpy.ndarray
        Test features for each fold with shape (n_samples, 1, ...)
    y_test : numpy.ndarray
        Test targets for each fold with shape (n_samples, 1, ...)
    """
    n_samples = len(X)

    # Initialize arrays to store all folds
    # The shape will be (n_samples, n_samples-1, ...) for train
    # and (n_samples, 1, ...) for test
    X_train = np.zeros((n_samples, n_samples - 1, *X.shape[1:]))
    y_train = np.zeros((n_samples, n_samples - 1, *y.shape[1:]))
    X_test = np.zeros((n_samples, 1, *X.shape[1:]))
    y_test = np.zeros((n_samples, 1, *y.shape[1:]))

    # Generate all possible splits
    for i in range(n_samples):
        # Create mask for training data
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False

        # Split data
        X_train[i] = X[mask]
        y_train[i] = y[mask]
        X_test[i] = X[~mask]
        y_test[i] = y[~mask]

    return X_train, y_train, X_test, y_test


def generate_batches_by_size(data_list, batch_size):
    """
    Splits a list into batches of a specified size.

    Args:
        data_list (list): The list to split into batches.
        batch_size (int): The size of each batch.

    Returns:
        list of lists: A list containing the batches as sublists.
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")

    # Generate batches using list slicing
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

    return batches


def generate_batches(data_list, num_batches):
    """
    Splits a list into a specified number of batches.

    Args:
        data_list (list): The list to split into batches.
        num_batches (int): The number of batches to create.

    Returns:
        list of lists: A list containing the batches as sublists.
    """
    if num_batches <= 0:
        raise ValueError("The number of batches must be greater than 0.")

    # Calculate the size of each batch (some might be larger if uneven division)
    avg_batch_size = len(data_list) // num_batches
    remainder = len(data_list) % num_batches

    batches = []
    start_index = 0

    for i in range(num_batches):
        # Add one extra item to the batch if we have a remainder
        batch_size = avg_batch_size + (1 if i < remainder else 0)
        batches.append(data_list[start_index:start_index + batch_size])
        start_index += batch_size

    return batches


class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, models, path, saver_freq):
        self.models = models
        self.path = path
        self.save_freq = saver_freq

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if (epoch % self.save_freq == 0):
            # print(f"Saving epoch: {epoch}, train_acc: {logs['acc']}, : {logs['batch_acc']}")
            for name in self.models:
                model = self.models[name]
                model.save(f"{self.path}/{name}.keras", overwrite=True)


@keras.saving.register_keras_serializable(package="saved_models")
class BinaryDense(keras.layers.Layer):
    def __init__(self, units, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(BinaryDense, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            name='kernel'
        )

        super(BinaryDense, self).build(input_shape)

    def call(self, inputs):
        # Binarize the kernel weights
        # kernel_binarized = keras.ops.sign(self.kernel)

        # Perform the matrix multiplication
        output = keras.ops.sign(keras.ops.matmul(inputs, self.kernel))

        return output

    def get_config(self):
        config = super(BinaryDense, self).get_config()
        config.update({
            'units': self.units,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
        })
        return config


# class OrthogonalConstraint(keras.constraints.Constraint):
#     def __call__(self, w):
#         #print(w.shape)
#         #exit()
#         # Perform Singular Value Decomposition
#         u, _, v = keras.ops.linalg.svd(w)
#         # Reconstruct the nearest orthogonal matrix
#         return keras.ops.matmul(u, keras.ops.transpose(v))

def average_maps(maps):
    # Initialize an empty dictionary to store the averages
    averaged_map = {}

    # Iterate over the keys of the first map (assuming all maps have the same keys)
    for key in maps[0]:
        # Sum the values for the current key from all maps
        total = sum(map[key] for map in maps)
        # Calculate the average
        average = total / len(maps)
        # Store the average in the new map
        averaged_map[key] = average

    return averaged_map


class NNWeightHelper:
    def __init__(self, model):
        self.model = model
        self.init_weights = self.model.get_weights()

    def _set_trainable_weight(self, model, weights):
        """Sets the weights of the model.

        # Arguments
            model: a keras neural network model
            weights: A list of Numpy arrays with shapes and types matching
                the output of `model.trainable_weights`.
        """

        # for sw, w in zip(layer.trainable_weights, weights):
        #      tuples.append((sw, w))

        model.set_weights(tuples)


def categorical_crossentropy(
        y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the categorical crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to `-1`. The dimension along which the entropy is
            computed.

    Returns:
        Categorical crossentropy loss value.

    Example:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.0513, 2.303], dtype=float32)
    """
    if isinstance(axis, bool):
        raise ValueError(
            "`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    if y_pred.shape[-1] == 1:
        warnings.warn(
            "In loss categorical_crossentropy, expected "
            "y_pred.shape to be (batch_size, num_classes) "
            f"with num_classes > 1. Received: y_pred.shape={y_pred.shape}. "
            "Consider using 'binary_crossentropy' if you only have 2 classes.",
            SyntaxWarning,
            stacklevel=2,
        )

    if label_smoothing:
        num_classes = ops.cast(ops.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (
                label_smoothing / num_classes
        )

    return ops.categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits, axis=axis
    )


def set_weights(self, weights):
    tuples = []

    for w in self.init_weights:
        num_param = w.size

        layer_weights = weights[:num_param]
        new_w = np.array(layer_weights).reshape(w.shape)
        # print(new_w.shape)

        tuples.append(new_w)
        weights = weights[num_param:]

    self.model.set_weights(tuples)


def get_weights(self):
    W_list = (self.model.trainable_weights)
    W_flattened_list = [np.array(k).flatten() for k in W_list]
    W = np.concatenate(W_flattened_list)

    return W

@keras.saving.register_keras_serializable(package="saved_models")
class OneHotLayer(keras.layers.Layer):
    def __init__(self, num_classes=11, axis=-1, name="oh", **kwargs):
        super(OneHotLayer, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.axis = axis

    def call(self, inputs):
        return keras.ops.one_hot(inputs, self.num_classes, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.insert(self.axis if self.axis >= 0 else len(output_shape) + 1 + self.axis,
                            self.num_classes)


        return tuple(output_shape)

    def get_config(self):
        config = super(OneHotLayer, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'axis': self.axis
        })
        return config


@keras.saving.register_keras_serializable(package="saved_models")
class EvenOddLayer(keras.layers.Layer):
    def __init__(self, max_elements=10, even=True, name="evenodd", **kwargs):
        super(EvenOddLayer, self).__init__(name=name, **kwargs)
        self.max_elements = max_elements
        arr = np.array(range(max_elements))
        self.odd_mask = arr % 2 != 0
        self.even_mask = arr % 2 == 0
        self.even = even

    def call(self, inputs):
        if self.even:
            return inputs[:,self.even_mask]
        else:
            return inputs[:,self.odd_mask]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        return (output_shape[0], output_shape[1]//2, *output_shape[2:])

    def get_config(self):
        config = super(EvenOddLayer, self).get_config()
        config.update({
            'even': self.even,
            'max_elements': self.max_elements
        })
        return config

import csv

def write_dict_to_csv(dictionary, csv_filename):
    """
    Writes a dictionary to a CSV file with the dictionary keys as the header.
    Appends to the file if it already exists and has a header.

    Parameters:
    dictionary (dict): The dictionary to write to the CSV file.
    csv_filename (str): The name of the CSV file to create or append to.
    """
    # Convert dictionary values to a row
    rows = [list(dictionary.values())]

    # Define the header based on dictionary keys
    header = list(dictionary.keys())

    # Check if the file already exists and has a header
    try:
        with open(csv_filename, mode='r') as file:
            has_header = file.readline().strip() == ",".join(header)
    except FileNotFoundError:
        has_header = False

    # Write to the CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not has_header:
            writer.writerow(header)
        writer.writerows(rows)


from sklearn.linear_model import LinearRegression

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from itertools import combinations

class MultiTaskL0LinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_significant_features):
        self.n_significant_features = n_significant_features

    def fit(self, X, y):
        # Validate inputs
        X, y = check_X_y(X, y, multi_output=True)
        self.n_samples_, self.n_features_ = X.shape
        self.n_tasks_ = y.shape[1]

        if self.n_significant_features > self.n_features_ + 1:  # Include intercept as a feature
            raise ValueError("n_significant_features cannot exceed the number of features in X plus one for the intercept.")

        self.selected_features_ = []
        self.regressors_ = []

        X = np.array(X,dtype="float")
        y = np.array(y, dtype="float")

        print("actions", X[:,-1])

        noise = np.random.random(size=(X.shape[0], X.shape[1]-1))*0.0000001

        #print(noise.shape, X.shape, y.shape)
        #exit()

        X[:,:-1] +=noise
        y +=noise

        # Add intercept as an additional column to X
        X_with_intercept = np.hstack([X, np.ones((self.n_samples_, 1))])

        # Perform exhaustive search for each task individually
        for task_idx in range(self.n_tasks_):
            y_task = y[:, task_idx]

            best_score = -np.inf
            best_features = None
            best_weights_penalty = np.inf

            # Iterate over possible numbers of features (from 1 to n_significant_features)
            for k in range(1, self.n_significant_features + 1):
                for feature_combination in combinations(range(self.n_features_ + 1), k):
                    selected_features = np.array(feature_combination)
                    X_reduced = X_with_intercept[:, selected_features]

                    # Fit a linear model on the reduced feature set for the task

                    # Fit a linear model on the reduced feature set for the task
                    # loo = LeaveOneOut()
                    # scores = []
                    # for train_index, test_index in loo.split(X_reduced):
                    #     # Split the data into train and test sets
                    #     X_train, X_test = X_reduced[train_index], X_reduced[test_index]
                    #     y_train, y_test = y_task[train_index], y_task[test_index]
                    #
                    #     # Train the model
                    #     regressor = LinearRegression(fit_intercept=False)
                    #     regressor.fit(X_train, y_train)
                    #
                    #     # Make predictions
                    #     y_pred = regressor.predict(X_test)
                    #
                    #     # Calculate the error (for example, Mean Squared Error)
                    #     error = mean_squared_error(y_test, y_pred)
                    #     scores.append(1.0 - error)

                    regressor = LinearRegression(fit_intercept=False)
                    regressor.fit(X_reduced, y_task)

                    # Evaluate model performance using R^2 score
                    #score = regressor.score(X_reduced, y_task)
                    pred = regressor.predict(X_reduced)
                    score = 1.0 -sklearn.metrics.mean_squared_error(y_task, pred)
                    #score = 1 -

                    # Compute a penalty based on how close the weights are to 1
                    weights_penalty = np.sum(np.abs(regressor.coef_ - 1))

                    #¡print(task_idx, k, regressor.coef_, feature_combination, score, X_reduced)

                    # Keep track of the best combination, prioritizing lower weights_penalty
                    if (score > best_score):
                        best_score = score
                        best_features = selected_features
                        best_weights_penalty = weights_penalty

            # Store the best features and refit the model for the task
            self.selected_features_.append(best_features)
            X_reduced = X_with_intercept[:, best_features]

            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(X_reduced, y_task)
            self.regressors_.append(regressor)

        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self, ["selected_features_", "regressors_"])

        # Add intercept as an additional column to X
        X_with_intercept = np.hstack([X, np.ones((X.shape[0], 1))])

        # Predict for each task individually
        predictions = []
        for task_idx, regressor in enumerate(self.regressors_):
            X_reduced = X_with_intercept[:, self.selected_features_[task_idx]]
            predictions.append(regressor.predict(X_reduced))

        return np.column_stack(predictions)

    def get_support(self):
        """Get the masks or indices of the selected features for each task."""
        check_is_fitted(self, "selected_features_")
        support = []
        for features in self.selected_features_:
            mask = np.zeros(self.n_features_ + 1, dtype=bool)  # Include intercept
            mask[features] = True
            support.append(mask)
        return support


import keras
from keras import ops


def s(x, epsilon=1e-30):
    return ops.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )


def s(x, epsilon=1e-30):
    return ops.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )


def log_stablemax(x, axis=-1):
    """
    Similar to log_softmax but using the stable s() function instead of exp()
    """
    s_x = s(x)
    return ops.log(s_x) - ops.log(ops.sum(s_x, axis=axis, keepdims=True))


def sparse_stablemax_crossentropy(y_true, y_pred, from_logits=True, axis=-1):
    """
    Sparse categorical crossentropy using stablemax instead of softmax.
    """
    y_pred = ops.cast(y_pred, 'float64')
    y_true = ops.cast(y_true, 'int32')

    if not from_logits:
        epsilon = 1e-30
        y_pred = ops.clip(y_pred, epsilon, 1 - epsilon)
        y_pred = ops.log(y_pred)

    log_probs = log_stablemax(y_pred, axis=axis)

    # Create a one-hot matrix and use it for reduction
    num_classes = ops.shape(log_probs)[-1]
    y_true_one_hot = ops.one_hot(y_true, num_classes)

    # Multiply and sum to get the log probs of the true classes
    prediction_log_probs = -ops.sum(log_probs * y_true_one_hot, axis=-1)

    return prediction_log_probs
class SparseCrossentropyStablemax(keras.losses.Loss):
    """
    Sparse Categorical Crossentropy loss using stablemax stabilization.
    """

    def __init__(
            self,
            from_logits=True,
            name="sparse_crossentropy_stablemax",
    ):
        super().__init__(name=name)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return sparse_stablemax_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )


import numpy as np
from collections import Counter
from typing import Tuple, List, Dict
from tqdm import tqdm
import multiprocessing as mp


def encode_colors(images: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Encode colors in multiple 2D images based on overall frequency.

    Args:
        images: numpy array of shape (32, 32, n) containing integer color codes

    Returns:
        Tuple containing:
        - numpy array of encoded images
        - decoder dictionary mapping new codes to original colors
    """
    # Flatten all images to count colors
    flat_colors = images.flatten()

    # Count unique colors across all images
    color_counts = Counter(flat_colors)

    # Sort colors by frequency (most frequent first)
    sorted_colors = sorted(color_counts.items(), key=lambda x: (-x[1], x[0]))

    # Create encoding and decoding mappings
    color_to_code = {color: i+1 for i, (color, _) in enumerate(sorted_colors)}
    code_to_color = {i: original_color for original_color, i in color_to_code.items()}

    # Create vectorized mapping function
    color_map = np.zeros(max(flat_colors) + 1, dtype=np.int32)
    for original_color, new_code in color_to_code.items():
        color_map[original_color] = new_code

    # Apply mapping to all images at once
    encoded_images = color_map[images]

    return encoded_images, code_to_color


def decode_colors(encoded_images: np.ndarray, decoder: dict) -> np.ndarray:
    """
    Decode encoded images back to original colors.

    Args:
        encoded_images: numpy array of shape (32, 32, n) containing encoded colors
        decoder: Dictionary mapping encoded values back to original colors

    Returns:
        numpy array of original color values
    """
    # Create reverse mapping array
    max_code = max(decoder.keys())
    color_map = np.zeros(max_code + 1, dtype=np.int32)
    for new_code, original_color in decoder.items():
        color_map[new_code] = original_color

    # Apply mapping to all images at once
    decoded_images = color_map[encoded_images]

    return decoded_images


def process_single_group(group_data: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray, dict]:
    """
    Process a single group of images. Helper function for multiprocessing.

    Args:
        group_data: Tuple containing (index, image_group)

    Returns:
        Tuple containing (index, encoded_images, decoder)
    """
    idx, group = group_data
    encoded_group, decoder = encode_colors(group)
    return idx, encoded_group, decoder


def process_image_batches(all_images: np.ndarray, num_processes: int = 50) -> Tuple[
    np.ndarray, List[Dict]]:
    """
    Process large batch of images in smaller groups using multiprocessing.

    Args:
        all_images: numpy array of shape (N, 32, 32, batch_size) where N is total number of groups
        batch_size: size of each batch (default 20)
        num_processes: number of processes to use (default: number of CPU cores)

    Returns:
        Tuple containing:
        - numpy array of encoded images with same shape as input
        - list of decoder dictionaries, one for each group of batch_size images
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    num_groups = all_images.shape[0]
    encoded_all = np.zeros_like(all_images)
    decoders = [None] * num_groups

    # Prepare data for multiprocessing
    group_data = [(i, all_images[i]) for i in range(num_groups)]

    # Create process pool and process groups
    with mp.Pool(processes=num_processes) as pool:
        # Process groups with progress bar
        results = list(tqdm(
            pool.imap(process_single_group, group_data),
            total=num_groups,
            desc=f"Processing image groups (using {num_processes} processes)"
        ))

    # Collect results
    for idx, encoded_group, decoder in results:
        encoded_all[idx] = encoded_group
        decoders[idx] = decoder

    return encoded_all, decoders


import matplotlib.pyplot as plt
from visualization.visualize_data import plot_data
def plot_images_with_action(x, a, y, y_prime, filename_id):
    """
    Plots three images (x, y, y') side-by-side with an action indicator in between.

    Args:
        x (numpy.ndarray): (32, 32) input image.
        a (int): Action indicator (0 or 1).
        y (numpy.ndarray): (32, 32) target image.
        y_prime (numpy.ndarray): (32, 32) predicted or alternate image.
        filename_id (str): Identifier for the filename when saving the plot.
    """

    # Create the plot
    fig, ax = plt.subplots(1, 4, figsize=(8, 2))

    extent = [-0.5, 31.5, -0.5, 31.5]

    # Plot the first image (x)
    #ax[0].imshow(x, cmap=cmap, interpolation='nearest')
    ax[0] = plot_data(x, extent=extent, axis = ax[0])
    ax[0].axis('off')
    ax[0].set_title("Image X")

    # Display the action in the second position
    ax[1].text(0.5, 0.5, str(a), fontsize=16, ha='center', va='center')
    ax[1].axis('off')
    ax[1].set_title("Action")

    # Plot the second image (y)
    ax[2] = plot_data(y, extent=extent, axis = ax[2])
    ax[2].axis('off')
    ax[2].set_title("Image Y")

    # Plot the third image (y')
    ax[3] = plot_data(y_prime, extent=extent, axis = ax[3])
    ax[3].axis('off')
    ax[3].set_title("Image Y'")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{filename_id}.png", bbox_inches='tight')
    plt.savefig(f"{filename_id}.svg", bbox_inches='tight')
    plt.close()

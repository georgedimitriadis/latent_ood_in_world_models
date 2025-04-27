import keras


def cce(y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1):
    return keras.metrics.categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing, axis)


def nrmse(y_true, y_pred):
    rmse = keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_pred - y_true), axis=0))

    # Calculate the range (max - min) of true values for each output column
    range_y_true = keras.ops.max(y_true, axis=0) - keras.ops.min(y_true, axis=0)

    # Normalize the RMSE by the range of each output column
    nrmse = rmse / (range_y_true + 0.00001)  # K.epsilon() avoids division by zero
    return nrmse

def logcosh(y_true, y_pred):
    logcosh = keras.ops.mean(keras.ops.log(keras.ops.cosh(y_pred - y_true)), axis=0)
    return logcosh

def r2(y_true, y_pred):


    #Residual sum of squares for each output
    ss_res = keras.ops.sum(keras.ops.square(y_true - y_pred), axis=0)

    # Total sum of squares for each output
    ss_tot = keras.ops.sum(keras.ops.square(y_true - keras.ops.mean(y_true, axis=0)), axis=0)

    # R^2 score for each output
    r2 = 1- ss_res / (ss_tot + 0.00000000000001)

    return -r2

def acc_seq(y_true, y_pred):
    return keras.ops.mean(keras.ops.min(keras.ops.equal(keras.ops.argmax(y_true, axis=-1),
                  keras.ops.argmax(y_pred, axis=-1)), axis=-1))




def b_acc2(y_true, y_pred):
    # Convert softmax probabilities to class predictions

    #print(y_pred.shape, y_true.shape)
    y_pred_classes = keras.ops.argmax(y_pred, axis=-1)
    y_true_classes = keras.ops.argmax(y_true, axis=-1)
    #print(y_pred_classes.shape, y_true_classes.shape)

    # Compare predictions with true values

    correct_pixels = keras.ops.equal(y_pred_classes, y_true_classes)

    #print(correct_pixels.shape)

    # Check if all pixels in an image are correct
    correct_images = keras.ops.all(correct_pixels, axis=[1, 2])
    #print(correct_images)
    correct_images = keras.ops.sum(keras.ops.cast(correct_images, "float32"))
    #print(correct_images.shape, "shape images")

    # Calculate the proportion of correct images in the batch
    #accuracy = keras.ops.mean(keras.ops.cast(correct_images, "float32"))

    return correct_images




def b_acc(y_true, y_pred):
    # Get the predicted class by taking the argmax along the last dimension (the class dimension)
    pred_classes = keras.ops.argmax(y_pred, axis=-1)

    # Get the true class (already one-hot encoded, so take argmax)
    true_classes = keras.ops.argmax(y_true, axis=-1)

    # Compare predicted classes to true classes for each sample
    correct_predictions = keras.ops.equal(pred_classes, true_classes)

    # Sum the incorrect predictions for each sample
    incorrect_predictions = keras.ops.sum(keras.ops.cast(~correct_predictions,
                                                      "float32"),
                                                 axis=[1, 2])

    # A sample is correct only if the sum of incorrect predictions is 0
    all_correct = keras.ops.cast(keras.ops.equal(incorrect_predictions, 0),
                                 "float32")

    # Calculate the percentage of samples that are fully correct
    accuracy = keras.ops.mean(keras.ops.cast(all_correct, "float32"))

    return accuracy

@keras.saving.register_keras_serializable(package="models.lm")
def b_acc_s(y_true, y_pred):
    # Get the predicted class by taking the argmax along the last dimension (the class dimension)
    pred_classes = keras.ops.argmax(y_pred, axis=-1)

    # Get the true class (already one-hot encoded, so take argmax)
    true_classes = y_true

    # Compare predicted classes to true classes for each sample
    correct_predictions = keras.ops.equal(pred_classes, true_classes)

    # Sum the incorrect predictions for each sample
    incorrect_predictions = keras.ops.sum(keras.ops.cast(~correct_predictions,
                                                      "float32"),
                                                 axis=[1, 2])

    # A sample is correct only if the sum of incorrect predictions is 0
    all_correct = keras.ops.cast(keras.ops.equal(incorrect_predictions, 0),
                                 "float32")

    # Calculate the percentage of samples that are fully correct
    accuracy = keras.ops.mean(keras.ops.cast(all_correct, "float32"))

    return accuracy




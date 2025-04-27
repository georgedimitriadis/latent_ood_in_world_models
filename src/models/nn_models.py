
import math

import keras
import keras_hub
from keras import layers

from models.nn_components import (build_encoder,
                                  build_decoder, Patches, PatchEncoder, mlp,
                                  AttentionMappingLayer, AttentionMappingLayer2D,
                                  EmergentSymbolBindingLayer, SmoothBlendLayer,
                                  gMLPLayer, MLPMixerLayer, FNetLayer, SpatialCopyLayer)
from models.utils import OneHotLayer


def cnn(input_shape=(32, 32, 11), base_filters=64, encoder_filters=128, n_tasks=2):
    # Create the saved_models
    input_z_shape = (1,)
    encoder_input = (32, 32, 11)

    s_input_x = layers.Input(shape=input_shape[:2], name="x_input")
    s_input_z = layers.Input(shape=input_z_shape, name="z_input")

    s_input_x_oh = OneHotLayer(num_classes=11, name="oh_x")(s_input_x)

    s_encoder_x = build_encoder(encoder_input, base_filters, encoder_filters, "gelu",
                                "s_encoder_x")

    encoded_x = s_encoder_x(s_input_x_oh)

    encoded_z = layers.Embedding(n_tasks, 64)(s_input_z)
    encoded_z = layers.Reshape([8, 8, 1])(encoded_z)

    x = layers.concatenate([encoded_x, encoded_z])

    sprime_decoder = build_decoder(x.shape[1:], 11,
                                   base_filters, "gelu")(x)
    sprime_decoder = layers.Activation("softmax")(sprime_decoder)
    model = keras.Model([s_input_x, s_input_z], sprime_decoder, name="lr")

    return model


def axial_pointer_network(input_shape=(32, 32, 11), patch_size=8,
                          with_language: str = False, with_mask: bool = False,
                          line_features: bool = False,
                          projection_dim=256, dropout=0.3,
                          n_mlp_layers=10, n_mlp_units=256,
                          n_tasks=2, emb_task_dimension=128,
                          n_vocabulary=200, word_dimension=256, n_words_per_sentence=10,
                          noise=0.05):
    # Create the saved_models
    input_z_shape = (1,)
    s_input_x = layers.Input(shape=input_shape[:2], name="x_input")
    s_input_z = layers.Input(shape=input_z_shape, name="z_input")

    s_input_x_oh = OneHotLayer(num_classes=input_shape[-1], name="oh_x")(s_input_x)

    patch_size = patch_size
    num_patches = (input_shape[0] // patch_size) ** 2

    pe_x = PatchEncoder(num_patches, projection_dim, None)

    mlp_copy = None

    height = input_shape[0]
    width = input_shape[1]
    if line_features:
        attention_logits = layers.Dense(math.prod(input_shape[:2]) * 2, name=f"attention_logits")
    else:
        attention_logits = layers.Dense((height + width) * height * width,  name=f"attention_logits")

    emb_z = layers.Embedding(n_tasks, emb_task_dimension, name="z_embeddings")
    if with_language:
        # For use in multiple words task descriptions
        esbl = EmergentSymbolBindingLayer(n_vocabulary, word_dimension, gumbel_softmax=True)
        os = []
        for i in range(n_words_per_sentence):
            o = layers.Flatten()(emb_z(s_input_z))
            o = layers.Dense(n_vocabulary, name=f"word_attention_{i}")(o)
            o = esbl(o)
            os.append(o)
    else:
        os = [layers.Flatten()(emb_z(s_input_z))]

    z = layers.add(os)

    if with_mask:
        mask_mlp = layers.Dense(math.prod(input_shape[:2]), name=f"layer_mask")
        new_image_mlp = layers.Dense(math.prod(input_shape), name=f"layer_new")

    n_atten_layers = 1
    canvas = s_input_x_oh
    start = layers.GaussianNoise(noise)(s_input_x_oh)
    for i in range(n_atten_layers):

        x = Patches(patch_size)(start)
        #print(x.shape)
        x = pe_x(x)
        for g in range(n_mlp_layers):
             x = gMLPLayer(num_patches, projection_dim, dropout,
                           name=f"gMLP_{g}")(x)

        x = layers.Flatten()(x)
        c = layers.concatenate([x, z])

        if mlp_copy is None:
            mlp_copy = mlp(c.shape[1:],
                           n_mlp_layers,
                           n_mlp_units,
                           dropout,
                           name=f"mlp")

        c_mlp = mlp_copy(c)

        if not line_features:
            copy_layer = SpatialCopyLayer(gumbel_softmax=True)
        else:
            copy_layer =AttentionMappingLayer2D(gumbel_softmax=True)

        atten_c = (copy_layer([canvas, attention_logits(c_mlp)]))

        output = atten_c

        # Create Mask in order to copy over pixels from a new image
        if with_mask:
            new_image_mask = mlp(c_mlp.shape[1:], n_mlp_layers,
                                 n_mlp_units, dropout, name=f"mlp_new_mask")(c_mlp)

            mask = new_image_mask
            mask = mask_mlp(mask)
            mask = layers.Reshape(input_shape[:2], name="mask")(mask)

            new_image = new_image_mask
            new_image = new_image_mlp(new_image)
            new_image = layers.Reshape(input_shape)(new_image)
            new_image = layers.Activation("softmax", name="new_image")(new_image)

            final = SmoothBlendLayer(name="smooth_blend")([new_image, atten_c, mask])
            output = final

    model = keras.Model([s_input_x, s_input_z], output, name="lr")

    return model


def transformer(input_shape=(32, 32, 11), patch_size=8, with_language: str = False,
                projection_dim=256, dropout=0.3,
                n_mlp_layers=10, n_mlp_units=256,
                n_tasks=2, emb_task_dimension=128,
                n_vocabulary=200, word_dimension=256, n_words_per_sentence=10,
                noise=0.05):
    # Create the saved_models
    input_z_shape = (1,)

    s_input_x = layers.Input(shape=input_shape[:2], name="x_input")
    s_input_x_prev = layers.Input(shape=input_shape[2:], name="s_input_x_prev")

    s_input_z = layers.Input(shape=input_z_shape, name="z_input")

    s_input_x_oh = OneHotLayer(num_classes=11, name="oh_x")(s_input_x)

    # print(canvas.shape)
    # exit()
    s_input_x_prev_oh = OneHotLayer(num_classes=11, name="oh_x_prev")(s_input_x_prev)

    x_prev = s_input_x_prev_oh

    patch_size = patch_size  # 4
    num_patches = (input_shape[0] // patch_size) ** 2

    pe_x = PatchEncoder(num_patches, projection_dim, None)

    mlp_copy = None

    attention_inputs = layers.Dense(input_shape[0] * input_shape[1] * 2)
    # attention_inputs_new = layers.Dense(input_shape[0] * input_shape[1] )

    # For use with single word task and no effort to learn the task description
    emb_z = layers.Embedding(n_tasks, projection_dim, name="z_embeddings")
    if with_language:
        esbl = EmergentSymbolBindingLayer(n_vocabulary, word_dimension, gumbel_softmax=False)
        os = []
        for i in range(n_words_per_sentence):
            o = layers.Flatten()(emb_z(s_input_z))
            o = layers.Dense(n_vocabulary, name=f"word_attention_{i}")(o)
            o = esbl(o)
            os.append(o)
        os = [keras.ops.expand_dims(layer, axis=1) for layer in os]
        z = layers.concatenate(os, axis=1)
    else:
        o = layers.Flatten()(emb_z(s_input_z))
        z = keras.ops.expand_dims(o, axis=1)

    start = layers.GaussianNoise(noise)(s_input_x_oh)

    x = Patches(patch_size)(start)
    x = pe_x(x)
    x = keras.ops.concatenate([x, z], axis=1)

    for _ in range(n_mlp_layers):
        x = keras_hub.layers.TransformerEncoder(n_mlp_units, 8, 0.1)(x)

    x = layers.GlobalAvgPool1D()(x)
    x = layers.Dense(32 * 32 * 11)(x)
    x = layers.Reshape([32, 32, 11])(x)
    x = layers.Activation("softmax")(x)

    final = x

    model = keras.Model([s_input_x, s_input_z], final, name="lr")

    return model


def mlp_nn(input_shape=(32, 32, 11), patch_size=8, with_language: str = False,
           projection_dim=256, dropout=0.3,
           n_mlp_layers=10, n_mlp_units=256,
           n_tasks=11, emb_task_dimension=128,
           n_vocabulary=200, word_dimension=176, n_words_per_sentence=10,
           noise=0.05):
    # Create the saved_models
    input_z_shape = (1,)

    s_input_x = layers.Input(shape=input_shape[:2], name="x_input")
    s_input_x_prev = layers.Input(shape=input_shape[2:], name="s_input_x_prev")

    s_input_z = layers.Input(shape=input_z_shape, name="z_input")

    s_input_x_oh = OneHotLayer(num_classes=11, name="oh_x")(s_input_x)

    patch_size = patch_size  # 4
    num_patches = (input_shape[0] // patch_size) ** 2

    pe_x = PatchEncoder(num_patches, projection_dim, None)

    # For use with single word task and no effort to learn the task description
    emb_z = layers.Embedding(n_tasks, projection_dim, name="z_embeddings")
    if with_language:
        esbl = EmergentSymbolBindingLayer(n_vocabulary, word_dimension, gumbel_softmax=True)
        os = []
        for i in range(n_words_per_sentence):
            o = layers.Flatten()(emb_z(s_input_z))
            o = layers.Dense(n_vocabulary, name=f"word_attention_{i}")(o)
            o = esbl(o)
            os.append(o)
        os = [keras.ops.expand_dims(layer, axis=1) for layer in os]
        z = layers.concatenate(os, axis=1)
    else:
        o = layers.Flatten()(emb_z(s_input_z))
        z = keras.ops.expand_dims(o, axis=1)

    start = layers.GaussianNoise(noise)(s_input_x_oh)

    x = Patches(patch_size)(start)
    x = pe_x(x)
    x = keras.ops.concatenate([x, z], axis=1)

    for _ in range(n_mlp_layers):
        x = FNetLayer(projection_dim, 0.1)(x)
        #print(x.shape)

    x = keras.layers.Flatten()(x)

    x = layers.Dense(32 * 32 * 11)(x)
    x = layers.Reshape([32, 32, 11])(x)
    x = layers.Activation("softmax")(x)

    final = x

    model = keras.Model([s_input_x, s_input_z], final, name="lr")

    return model



import keras
from keras import layers, models
from keras import ops

normalisation = layers.BatchNormalization
# regulariser = keras.regularizers.l2(0.001)
regulariser = None
constraint = None
constraint = keras.constraints.MaxNorm(0.1)


def residual_block(x, filters, activation, stride=1, transpose=False, ):
    conv = layers.Conv2DTranspose if transpose else layers.Conv2D

    y = conv(filters, kernel_size=3, strides=stride, padding='same',
             kernel_regularizer=regulariser, kernel_constraint=constraint)(x)
    y = normalisation()(y)
    y = layers.Activation(activation)(y)
    # y = layers.GaussianNoise(0.01)(y)

    y = conv(filters, kernel_size=3, strides=1, padding='same',
             kernel_regularizer=regulariser, kernel_constraint=constraint)(y)
    y = normalisation()(y)
    # y = layers.GaussianNoise(0.01)(y)

    if stride != 1 or x.shape[-1] != filters:
        x = conv(filters, kernel_size=1, strides=stride, padding='same',
                 kernel_regularizer=regulariser, kernel_constraint=constraint)(x)
        x = normalisation()(x)
        # x = layers.GaussianNoise(0.01)(x)

    y = layers.Add()([x, y])
    y = layers.Activation(activation)(y)

    return y


def build_dense_block(input_shape, filters, activation, num_layers, stride=1, transpose=False, name=None):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i in range(num_layers):
        current_stride = stride if i == 0 else 1
        x = residual_block(x, filters, activation, stride=current_stride, transpose=transpose)
        # print(x.shape, "x.shape", transpose)
    return models.Model(inputs, x, name=name)


def build_encoder(input_shape, base_filters, encoder_filters, activation, name="encoder"):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(base_filters, kernel_size=3, strides=1, padding='same',
                      kernel_regularizer=regulariser, kernel_constraint=constraint)(inputs)
    x = normalisation()(x)
    x = layers.Activation(activation)(x)

    # Stage 1
    stage1 = build_dense_block(x.shape[1:], base_filters, activation, num_layers=2, stride=1, name="encoder_stage1")
    x = stage1(x)

    # Stage 2
    stage2 = build_dense_block(x.shape[1:], base_filters * 2, activation, num_layers=2, stride=2, name="encoder_stage2")
    x = stage2(x)

    # Stage 3
    stage3 = build_dense_block(x.shape[1:], base_filters * 4, activation, num_layers=2, stride=2, name="encoder_stage3")
    x = stage3(x)

    encoded = layers.Conv2D(encoder_filters, kernel_size=3, strides=1,
                            padding='same', kernel_regularizer=regulariser, kernel_constraint=constraint)(x)
    encoded = normalisation()(encoded)
    encoded = layers.Activation(activation)(encoded)

    encoder = models.Model(inputs, encoded, name=name)
    return encoder


def build_decoder(input_shape, output_channels, base_filters, activation):
    decoder_input = layers.Input(shape=input_shape)
    x = decoder_input

    # Stage 3
    stage3 = build_dense_block(x.shape[1:], base_filters * 4, activation, num_layers=2, stride=2, transpose=True,
                               name="decoder_stage3")
    x = stage3(x)

    # Stage 2
    stage2 = build_dense_block(x.shape[1:], base_filters * 2, activation, num_layers=2, stride=2, transpose=True,
                               name="decoder_stage2")
    x = stage2(x)

    # Stage 1
    stage1 = build_dense_block(x.shape[1:], base_filters, activation, num_layers=2, stride=1, transpose=True,
                               name="decoder_stage1")
    x = stage1(x)

    # Final convolution
    decoded = layers.Conv2DTranspose(output_channels, kernel_size=3, strides=1,
                                     padding='same', kernel_regularizer=regulariser, kernel_constraint=constraint)(x)

    decoder = models.Model(decoder_input, decoded, name="decoder")
    return decoder

@keras.saving.register_keras_serializable(package="MyLayers")
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        patches = keras.ops.image.extract_patches(x, self.patch_size)
        batch_size = keras.ops.shape(patches)[0]
        num_patches = keras.ops.shape(patches)[1] * keras.ops.shape(patches)[2]
        patch_dim = keras.ops.shape(patches)[3]
        out = keras.ops.reshape(patches, (batch_size, num_patches, patch_dim))
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


def mlp(input_shape, n_layers, units, dropout, activation="gelu",
        name="mlp", kernel_constraint=None):
    inputs = layers.Input(shape=input_shape)

    norm = layers.BatchNormalization
    x = layers.Dense(units,
                     kernel_constraint=kernel_constraint)(inputs)
    x = norm()(x)

    previous_layers = [x]

    for i in range(n_layers):
        # Dense layer
        x = layers.Dense(units, use_bias=False,
                         kernel_constraint=kernel_constraint)(
            previous_layers[-1])  # No bias as BatchNorm handles it

        # Batch Normalization
        x = norm()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Activation(activation)(x)

        # Add a skip connection from the input to this layer
        if i > 0:  # Skip connection from earlier layers to the current one
            skip_connection = layers.Add()([x, previous_layers[0]])
        else:
            skip_connection = x  # No skip connection for the first layer

        # Append the skip connection to the list of layers
        previous_layers.append(skip_connection)

    x = previous_layers[-1]

    model = models.Model(inputs, x, name=name)

    return model


@keras.saving.register_keras_serializable(package="MyLayers")
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, weight_norm, **kwargs):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.weight_norm = weight_norm
        self.projection = layers.Dense(units=projection_dim, kernel_constraint=weight_norm)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim, embeddings_constraint=weight_norm
        )

    def call(self, patch):
        positions = keras.ops.expand_dims(
            keras.ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
                "weight_norm": self.weight_norm,
            }
        )
        return config


class AttentionMappingLayer(keras.Layer):
    def __init__(self, **kwargs):
        super(AttentionMappingLayer, self).__init__(**kwargs)
        self.temperature = 1.0
        self.seed_generator = keras.random.SeedGenerator(seed=1337)

    def build(self, input_shape):
        # Ensure inputs have the correct shape: (image, attention_weights)
        if len(input_shape) != 2:
            raise ValueError("Input must be a tuple of (image, attention_weights)")

        image_shape, attention_shape = input_shape

        # Ensure image has shape (batch_size, height, width, channels)
        if len(image_shape) != 4:
            raise ValueError("Image shape must be (batch_size, height, width, channels)")

        # Ensure attention weights have shape (batch_size, height * width, height * width)
        if len(attention_shape) != 3:
            raise ValueError("Attention weights shape must be (batch_size, height * width, height * width)")

        self.height = image_shape[1]
        self.width = image_shape[2]

    def call(self, inputs, training=None):
        # Unpack inputs: image and attention_weights
        image, logits = inputs

        if (training):

            u = keras.random.uniform(logits.shape, seed=self.seed_generator)
            gumbel = -ops.log(-ops.log(u + 1e-10) + 1e-10)

            # Add noise to logits
            noisy_logits = (logits + gumbel) / self.temperature

            # Softmax for differentiability during training
            attention_weights = ops.softmax(noisy_logits)
        else:
            index = ops.argmax(logits, axis=-1)
            attention_weights = ops.one_hot(index, logits.shape[-1])

        # Flatten the spatial dimensions of the input image
        batch_size = ops.shape(image)[0]
        flattened_image = ops.reshape(image, (batch_size, self.height * self.width, -1))

        # Apply the externally provided attention weights
        transformed = ops.matmul(attention_weights, flattened_image)  # Shape: (batch_size, height * width, channels)

        # Reshape back to image format
        output = ops.reshape(transformed, (batch_size, self.height, self.width, -1))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.saving.register_keras_serializable(package="MyLayers")
class EmergentSymbolBindingLayer(keras.Layer):
    def __init__(self, vocabulary_size, word_dimension, gumbel_softmax, **kwargs):
        super(EmergentSymbolBindingLayer, self).__init__(**kwargs)
        self.temperature = 1.0
        self.seed_generator = keras.random.SeedGenerator(seed=1337)
        self.word_dimension = word_dimension
        self.vocabulary_size = vocabulary_size
        self.initializer = "glorot_uniform"
        self.gumbel_softmax = gumbel_softmax

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=[1, self.vocabulary_size, self.word_dimension],
            initializer=self.initializer,
            trainable=True,
        )

    def call(self, logits, training=None):
        # Unpack inputs: image and attention_weights
        logits = ops.reshape(logits, [logits.shape[0], 1, logits.shape[1]])

        if training:
            if self.gumbel_softmax:
                u = keras.random.uniform(logits.shape, seed=self.seed_generator)
                gumbel = -ops.log(-ops.log(u + 1e-10) + 1e-10)
                noisy_logits = (logits + gumbel) / self.temperature
                attention_weights = ops.softmax(noisy_logits)
            else:
                attention_weights = keras.ops.softmax(logits)

        else:
            index = ops.argmax(logits, axis=-1)
            attention_weights = ops.one_hot(index, logits.shape[-1])

        batch_size = ops.shape(logits)[0]
        embs = ops.repeat(self.embeddings, batch_size, axis=0)

        transformed = ops.matmul(attention_weights, embs)
        transformed = ops.reshape(transformed, [transformed.shape[0], transformed.shape[2]])

        return transformed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "word_dimension": self.word_dimension,
                "vocabulary_size": self.vocabulary_size,
                "gumbel_softmax": self.gumbel_softmax,
            }
        )
        return config


class PositionEmbedding(keras.layers.Layer):
    def __init__(
            self,
            sequence_length,
            initializer="glorot_uniform",
            **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = keras.ops.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        # trim to match the length of the input sequence, which might be less
        # than the sequence_length of the layer.
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return keras.ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape


@keras.saving.register_keras_serializable(package="MyLayers")
class AttentionMappingLayer2D(keras.Layer):
    def __init__(self, gumbel_softmax, **kwargs):
        super(AttentionMappingLayer2D, self).__init__(**kwargs)
        self.temperature = 1.0
        self.seed_generator = keras.random.SeedGenerator(seed=1337)
        self.gumbel_softmax = gumbel_softmax

    def build(self, input_shape):
        # Ensure inputs have the correct shape: (image, attention_weights)
        if len(input_shape) != 2:
            raise ValueError("Input must be a tuple of (image, attention_weights)")

    def call(self, inputs, training=None):
        # Unpack inputs: image and attention_weights
        """
        Apply 2D attention with improved gradient flow by maintaining skip connections
        and proper normalization.
        """
        image, attention_logits = inputs

        batch_size = ops.shape(image)[0]
        height = ops.shape(image)[1]
        width = ops.shape(image)[2]
        num_channels = ops.shape(image)[-1]

        # Split attention logits
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

        if training:
            if self.gumbel_softmax:
                u = keras.random.uniform(attention_weights_h.shape, seed=self.seed_generator)
                gumbel = -ops.log(-ops.log(u + 1e-10) + 1e-10)
                noisy_logits = (attention_weights_h + gumbel) / self.temperature
                attention_weights_h = ops.softmax(noisy_logits)

                u = keras.random.uniform(attention_weights_w.shape, seed=self.seed_generator)
                gumbel = -ops.log(-ops.log(u + 1e-10) + 1e-10)
                noisy_logits = (attention_weights_w + gumbel) / self.temperature
                attention_weights_w = ops.softmax(noisy_logits)

            else:
                attention_weights_h = keras.ops.softmax(attention_weights_h)
                attention_weights_w = keras.ops.softmax(attention_weights_w)
        else:
            attention_weights_h = ops.argmax(attention_weights_h, axis=-1)
            attention_weights_h = ops.one_hot(attention_weights_h,
                                              attention_weights_h.shape[-1])

            attention_weights_w = ops.argmax(attention_weights_w, axis=-1)
            attention_weights_w = ops.one_hot(attention_weights_w,
                                              attention_weights_w.shape[-1])

        # === Height-wise Attention ===
        image_h = ops.reshape(image, (batch_size, height, width * num_channels))  # Merge width & channels
        transformed_h = ops.matmul(attention_weights_h, image_h)  # Apply attention along height
        transformed_h = ops.reshape(transformed_h, (batch_size, width, width, num_channels))

        # === Width-wise Attention ===
        image_w = ops.transpose(transformed_h, (0, 2, 1, 3))  # Swap height and width
        image_w = ops.reshape(image_w, (batch_size, width, width * num_channels))  # Merge height & channels
        transformed_w = ops.matmul(attention_weights_w, image_w)  # Apply attention along width
        transformed_w = ops.reshape(transformed_w, (batch_size, width, height, num_channels))
        transformed_w = ops.transpose(transformed_w, (0, 2, 1, 3))  # Swap height and width back

        output = transformed_w

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class SpatialCopyLayer(keras.Layer):
    def __init__(self, gumbel_softmax, **kwargs):
        """
        A layer that copies pixels from source positions to target positions in an image.

        Args:
            temperature: Temperature parameter for the softmax operation.
                         Lower values make the softmax more peaked (more deterministic).
        """
        super().__init__(**kwargs)
        self.temperature = 1.0
        self.seed_generator = keras.random.SeedGenerator(seed=1337)
        self.gumbel_softmax = gumbel_softmax

    def build(self, input_shape):
        # input_shape[0] is the image tensor: [batch_size, height, width, channels]
        # input_shape[1] is the logits tensor: [batch_size, height, width, height, width]
        self.image_shape = input_shape[0]
        self.logits_shape = input_shape[1]
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Applies axial attention for spatial copying using flattened logits.

        Args:
            inputs: A list containing [image_tensor, flattened_logits]
                - image_tensor: [batch_size, height, width, channels]
                - flattened_logits: [batch_size, (height + width) * height * width]
                  This contains both horizontal and vertical attention logits concatenated

        Returns:
            A tensor with the same shape as the input image.
        """
        image_tensor, flattened_logits = inputs
        batch_size = ops.shape(image_tensor)[0]
        height, width = self.image_shape[1], self.image_shape[2]

        # Calculate sizes for reshaping
        h_logits_size = height * width * height  # For each target (h,w), we have 'height' source row positions

        # Split the flattened logits into horizontal and vertical components
        h_logits_flat = flattened_logits[:, :h_logits_size]
        w_logits_flat = flattened_logits[:, h_logits_size:]

        # Reshape to proper dimensions
        h_logits = ops.reshape(h_logits_flat, [batch_size, height, width, height])  # [b, h, w, h]
        w_logits = ops.reshape(w_logits_flat, [batch_size, height, width, width])  # [b, h, w, w]

        # Apply softmax separately for horizontal and vertical attention
        if training:
            if self.gumbel_softmax:
                u = keras.random.uniform(h_logits.shape, seed=self.seed_generator)
                gumbel = -ops.log(-ops.log(u + 1e-10) + 1e-10)
                noisy_logits = (h_logits + gumbel) / self.temperature
                h_probs = ops.softmax(noisy_logits)

                u = keras.random.uniform(w_logits.shape, seed=self.seed_generator)
                gumbel = -ops.log(-ops.log(u + 1e-10) + 1e-10)
                noisy_logits = (w_logits + gumbel) / self.temperature
                w_probs = ops.softmax(noisy_logits)

            else:
                h_probs = ops.softmax(h_logits / self.temperature, axis=-1)  # [b, h, w, h]
                w_probs = ops.softmax(w_logits / self.temperature, axis=-1)  # [b, h, w, w]
        else:
            h_probs = ops.argmax(h_logits, axis=-1)
            h_probs = ops.one_hot(h_probs, h_probs.shape[-1])

            w_probs = ops.argmax(w_logits, axis=-1)
            w_probs = ops.one_hot(w_probs, w_probs.shape[-1])

        # First apply horizontal attention (for each target pixel, attend to pixels in the same column)
        h_gathered = ops.einsum('biwh,bhwc->biwc', h_probs, image_tensor)

        # Then apply vertical attention (for each target pixel with horizontal context, attend to pixels in the same row)
        output = ops.einsum('bijw,biwc->bijc', w_probs, h_gathered)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.saving.register_keras_serializable(package="MyLayers")
class SmoothBlendLayer(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Unpack the inputs
        image1, image2, mask = inputs

        # Ensure mask has same shape as the last channel
        mask = ops.expand_dims(mask, axis=-1)
        mask = ops.repeat(mask, 11, axis=-1)

        # Apply sigmoid with temperature for smoother transition
        # Higher temperature makes the transition sharper
        # Lower temperature makes it smoother
        blend_weights = ops.sigmoid(mask)



        # Smoothly blend between the two images
        output = image1 * (1 - blend_weights) + image2 * blend_weights

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class FNetLayer(layers.Layer):
    def __init__(self, embedding_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ffn = keras.Sequential(
            [
                layers.Dense(units=embedding_dim, activation="gelu"),
                layers.Dropout(rate=dropout_rate),
                layers.Dense(units=embedding_dim),
            ]
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply fourier transformations.
        real_part = inputs
        im_part = keras.ops.zeros_like(inputs)
        x = keras.ops.fft2((real_part, im_part))[0]
        # Add skip connection.
        x = x + inputs
        # Apply layer normalization.
        x = self.normalize1(x)
        # Apply Feedfowrad network.
        x_ffn = self.ffn(x)
        # Add skip connection.
        x = x + x_ffn
        # Apply layer normalization.
        return self.normalize2(x)


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = keras.ops.transpose(x, axes=(0, 2, 1))
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = keras.ops.transpose(mlp1_outputs, axes=(0, 2, 1))
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


@keras.saving.register_keras_serializable(package="MyLayers")
class gMLPLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_projection1 = keras.Sequential(
            [
                layers.Dense(units=embedding_dim * 2, activation="gelu"),
                layers.Dropout(rate=dropout_rate),
            ]
        )

        self.channel_projection2 = layers.Dense(units=embedding_dim)

        self.spatial_projection = layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def spatial_gating_unit(self, x):
        # Split x along the channel dimensions.
        # Tensors u and v will in the shape of [batch_size, num_patchs, embedding_dim].
        u, v = keras.ops.split(x, indices_or_sections=2, axis=2)
        # Apply layer normalization.
        v = self.normalize2(v)
        # Apply spatial projection.
        v_channels = keras.ops.transpose(v, axes=(0, 2, 1))
        v_projected = self.spatial_projection(v_channels)
        v_projected = keras.ops.transpose(v_projected, axes=(0, 2, 1))
        # Apply element-wise multiplication.
        return u * v_projected

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize1(inputs)
        # Apply the first channel projection. x_projected shape: [batch_size, num_patches, embedding_dim * 2].
        x_projected = self.channel_projection1(x)
        # Apply the spatial gating unit. x_spatial shape: [batch_size, num_patches, embedding_dim].
        x_spatial = self.spatial_gating_unit(x_projected)
        # Apply the second channel projection. x_projected shape: [batch_size, num_patches, embedding_dim].
        x_projected = self.channel_projection2(x_spatial)
        # Add skip connection.
        return x + x_projected

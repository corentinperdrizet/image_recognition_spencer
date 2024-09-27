############################################
####        Building the model          ####
############################################

import tensorflow as tf
from tensorflow.keras import layers, models, utils, preprocessing

def conv_block(input_tensor, num_filters):
    """Bloc de convolution pour U-Net."""
    x = layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(input_tensor, num_filters):
    """Bloc encodeur avec max pooling pour U-Net."""
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    """Bloc décodeur pour U-Net."""
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)
    x = layers.concatenate([x, concat_tensor])
    x = conv_block(x, num_filters)
    return x

def build_smallest_unet(input_shape):
    """Building the U-net model"""
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x1, p1 = encoder_block(inputs, 16)
    x2, p2 = encoder_block(p1, 32)
    x3, p3 = encoder_block(p2, 64)

    # Bridge
    bridge = conv_block(p3, 128)

    # Decoder
    d2 = decoder_block(bridge, x3, 64)
    d3 = decoder_block(d2, x2, 32)
    d4 = decoder_block(d3, x1, 16)

    # Output
    outputs = layers.Conv2D(6, (1, 1), activation="softmax")(d4)


    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Summary of the model
model = build_smallest_unet((512, 512, 3))
model.summary()

# Uncomment bellow for a visual representation of the model

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# Uncomment bellow for visualizing the layers of the model

# import visualkeras
# visualkeras.layered_view(model).show()
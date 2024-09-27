""" Display activations of layers in the U-Net model """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import Model
from config import unet_model
from utils.data_preprocessing import SatelliteDataGenerator, get_image_mask_paths

train_image_paths, train_mask_paths = get_image_mask_paths('data/train/images', 'data/train/masks')
val_image_paths, val_mask_paths = get_image_mask_paths('data/val/images', 'data/val/masks')

# Setup data generators
batch_size = 4
train_gen = SatelliteDataGenerator(train_image_paths, train_mask_paths, batch_size)
val_gen = SatelliteDataGenerator(val_image_paths, val_mask_paths, batch_size)

def display_model_activations(model, image, layer_names=None):
    if layer_names is not None:
        layers_outputs = [model.get_layer(name).output for name in layer_names]
    else:
        layers_outputs = [layer.output for layer in model.layers]  # Prendre toutes les couches
    
    activation_model = Model(inputs=model.input, outputs=layers_outputs)
    activations = activation_model.predict(np.expand_dims(image, axis=0))

    for layer_name, activation_map in zip(layer_names, activations):
        num_filters = activation_map.shape[-1]
        size = activation_map.shape[1]
        n_cols = 8
        n_rows = num_filters // n_cols
        display_grid = np.zeros((size * n_rows, n_cols * size))

        for col in range(n_cols):
            for row in range(n_rows):
                filter_img = activation_map[0, :, :, col * n_rows + row]
                filter_img -= filter_img.mean()
                filter_img /= filter_img.std()
                filter_img *= 64
                filter_img += 128
                filter_img = np.clip(filter_img, 0, 255).astype('uint8')
                display_grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = filter_img

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    plt.show()

display_model_activations(unet_model, val_gen[0][0][1], ['activation_9', 'activation_11', 'concatenate_5'])

""" Display the prediction of the model on the validation set 
Display the original image, the real mask and the predicted mask """
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import Model
import sys

# Add the project root to the path
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)
project_root = os.path.dirname(parent_directory)
sys.path.append(project_root)

from config import color_to_class, model
from utils.data_preprocessing import SatelliteDataGenerator, get_image_mask_paths

# Reverse dictionary to index by class
class_to_color = {v: np.array(k)/255.0 for k, v in color_to_class.items()}

# Name of the classes for the legend
class_names = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled']

# Create legend patches
patches = [mpatches.Patch(color=class_to_color[i], label=class_names[i]) for i in range(len(class_names))]

def display_image_and_pred(img, predicted_mask):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img.astype('uint8'))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    predicted_color_mask = np.zeros((*predicted_mask.shape, 3))
    for c in class_to_color:
        predicted_color_mask[predicted_mask == c] = class_to_color[c]
    plt.imshow(predicted_color_mask)
    plt.title('Predicted Mask')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()


def display_image_and_mask(img, real_mask, predicted_mask):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img.astype('uint8'))
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    real_color_mask = np.zeros((*real_mask.shape, 3))
    for c in class_to_color:
        real_color_mask[real_mask == c] = class_to_color[c]
    plt.imshow(real_color_mask)
    plt.title('Real Mask')

    plt.subplot(1, 3, 3)
    predicted_color_mask = np.zeros((*predicted_mask.shape, 3))
    for c in class_to_color:
        predicted_color_mask[predicted_mask == c] = class_to_color[c]
    plt.imshow(predicted_color_mask)
    plt.title('Predicted Mask')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

train_image_paths, train_mask_paths = get_image_mask_paths('data/train/images', 'data/train/masks')
val_image_paths, val_mask_paths = get_image_mask_paths('data/val/images', 'data/val/masks')

# Setup data generators
batch_size = 4
train_gen = SatelliteDataGenerator(train_image_paths, train_mask_paths, batch_size)
val_gen = SatelliteDataGenerator(val_image_paths, val_mask_paths, batch_size)

def display_multiple_images(n_images):
    for i in range(n_images):
        test_img, test_mask = val_gen[i]  # Get the ith validation batch
        predicted_mask = model.predict(test_img)
        predicted_mask = np.argmax(predicted_mask, axis=-1)  # Convert probabilities to classes

        for j in range(len(test_img)):  # Display each image in the batch
            display_image_and_mask(test_img[j], np.squeeze(test_mask[j]), predicted_mask[j])

display_multiple_images(3)
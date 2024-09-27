# Description: This file contains the functions to do two things:
#
# First is predicting the class of an image and 
# show the probability of beloning to each class
#  
# Second is predicting the class of all images in a folder

import os
import sys

# Add the project root to the path
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)
project_root = os.path.dirname(parent_directory)
sys.path.append(project_root)

from config import model, class_labels
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


# Path to an image to test
img_path = 'path/to/an/image'
# Path to the folder containing the images to test
test_folder = 'path/to/a/folder'


# Function to predict the probability of each class and display
def predict_image_with_plot(img_path, model, class_labels):
    # Load the image and preprocess it
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch of 1
    img_array /= 255.0  # Rescalation as in training

    # Predict the class of the image
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    
    # Display the image and the probabilities
    plt.figure(figsize=(10, 4))

    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image.load_img(img_path))
    plt.title(f"Prediction : {predicted_class}")
    plt.axis('off')  # Pas d'axes pour l'image

    # Display the probabilities
    plt.subplot(1, 2, 2)
    plt.bar(class_labels, predictions[0], color='blue')
    plt.title("Probability of each class")
    plt.ylabel('Probability')
    plt.xlabel('Class')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    return img, predicted_class, os.path.basename(img_path)


# Testing an image
predict_image_with_plot(img_path, model, class_labels)


# Testing a folder 

# Function to predict the class of an image
def predict_image(img_path, model, class_labels):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch of 1
    img_array /= 255.0  # Rescale as during training

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    img_loaded = image.load_img(img_path)

    return img_loaded, predicted_class, os.path.basename(img_path)


# Function to predict all images in a folder and display them in a grid
def predict_images_in_folder(folder_path, model, class_labels, rows=5, cols=4):
    """Function to predict all images in a folder and display them in specified grids."""
    images = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(images)
    images_per_grid = rows * cols
    num_grids = (num_images + images_per_grid - 1) // images_per_grid 

    image_idx = 0
    for grid in range(num_grids):
        plt.figure(figsize=(cols * 4, rows * 4))
        for i in range(images_per_grid):
            if image_idx >= num_images:
                break
            img_path = images[image_idx]
            img_loaded, predicted_class, img_name = predict_image(img_path, model, class_labels)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img_loaded)
            plt.title(f'{img_name}\nPrédiction : {predicted_class}', fontsize=8)
            plt.axis('off')
            image_idx += 1
        plt.subplots_adjust(wspace=0.5, hspace=0.6)
        plt.show()


# Test the images in the folder
predict_images_in_folder(test_folder, model, class_labels)


import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import sys

# Add the project root to the path
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)
project_root = os.path.dirname(parent_directory)
sys.path.append(project_root)

from config import model, mlb

def parse_json_labels(file_path):
    """ Parse tags from a JSON file and return them as a list. """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['labels']

# Load an RGB image from the specific bands
def load_rgb_image(folder_path):
    bands = ['B04', 'B03', 'B02']
    images = []
    for band in bands:
        file_path = os.path.join(folder_path, f"{folder_path.split('/')[-1]}_{band}.tif")
        with Image.open(file_path) as img:
            images.append(np.array(img))

    image_rgb = np.dstack(images)
    image_rgb = image_rgb / image_rgb.max()
    return image_rgb

# Predict classes from model
def predict_and_display(model, folder_path, mlb):
    images = load_images_from_folder(folder_path)
    inputs = prepare_inputs(images)
    predictions = model.predict(inputs)
    predicted_labels = (predictions > 0.5).astype(int)

    image_rgb = load_rgb_image(folder_path)
    
    label_path = os.path.join(folder_path, f"{folder_path.split('/')[-1]}_labels_metadata.json")
    real_labels = parse_json_labels(label_path)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Display RGB image
    axs[0].imshow(image_rgb)
    axs[0].set_title("RGB Image")
    axs[0].axis('off')  # Disable axes for the image

    # Display probabilities for each class
    classes = mlb.classes_
    bar_positions = np.arange(len(classes))
    axs[1].barh(bar_positions, predictions[0], align='center', color='blue', alpha=0.6)
    axs[1].set_yticks(bar_positions)
    axs[1].set_yticklabels(classes)
    axs[1].set_xlabel('Probabilities')
    axs[1].set_title('Class Predictions')

    # Mark real classes in red
    real_label_indices = [mlb.classes_.index(lbl) for lbl in real_labels if lbl in mlb.classes_]
    for idx in real_label_indices:
        axs[1].get_yticklabels()[idx].set_color('red')
    
    plt.tight_layout()
    plt.show()

    print("Classes predicted with probability > 50%:")
    for i, (label, score) in enumerate(zip(classes, predictions[0])):
        if score > 0.5:
            print(f"{label}: {score:.2f}")

# Load all images from a folder for prediction
def load_images_from_folder(folder_path):
    band_files = {'B01': None, 'B02': None, 'B03': None, 'B04': None, 'B05': None, 'B06': None,
                  'B07': None, 'B08': None, 'B09': None, 'B11': None, 'B12': None, 'B8A': None}
    for filename in os.listdir(folder_path):
        for band in band_files.keys():
            if band in filename:
                band_files[band] = load_image(os.path.join(folder_path, filename))
    return band_files

# Load and preprocess a single image
def load_image(file_path):
    with Image.open(file_path) as img:
        image_array = np.array(img)
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=-1)
        image_tensor /= 255.0
    return image_tensor.numpy()

def prepare_inputs(images):
    # Prepare inputs for different resolutions
    img_20 = np.stack([images['B01'], images['B09']], axis=-1)  # Stack 20m bands
    img_60 = np.stack([images['B05'], images['B06'], images['B07'], images['B8A'], images['B11'], images['B12']], axis=-1)  # Stack 60m bands
    img_120 = np.stack([images['B02'], images['B03'], images['B04'], images['B08']], axis=-1)  # Stack 120m bands

    # Remove unnecessary dimension
    input_20 = np.expand_dims(np.squeeze(img_20), axis=0)
    input_60 = np.expand_dims(np.squeeze(img_60), axis=0)
    input_120 = np.expand_dims(np.squeeze(img_120), axis=0)

    return [input_20, input_60, input_120]

# Function to evaluate a set of directories
def evaluate_directory(directory_path, model, mlb):
    for folder in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder)
        if os.path.isdir(folder_path):
            print(f"Processing image: {folder}")
            predict_and_display(model, folder_path, mlb)

# Main
if __name__ == "__main__":
    directory_path = '/Users/corentinperdrizet/Documents/internship/img_reco/bigearth/images_provider'
    evaluate_directory(directory_path, model, mlb)

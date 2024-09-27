# data_preprocessing.py
"""Descritpion: This script contains functions to preprocess the BigEarthNet dataset and store it in an HDF5 file.
 Indeed, the dataset is really huge and it is not possible to load it all in memory.
 So execute this file if you want to train the model on the BigEarthNet dataset and
 you still don't have the HDF5 files."""

import os
import json
import numpy as np
import h5py
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf

def parse_json_labels(file_path):
    """ Parse labels from a JSON file and return them as a list. """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['labels']

def load_image(file_path):
    """ Load and preprocess an image, including TIFF files. """
    image = Image.open(file_path)
    image = image.convert('L')
    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=-1)
    image_tensor /= 255.0
    return image_tensor.numpy()  # Convert to numpy array for compatibility with h5py

def load_images_from_folder(folder_path):
    """ Load all TIFF images from a folder into a dictionary. """
    images = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.tif'):
            band_name = filename.split('_')[-1][:-4]
            file_path = os.path.join(folder_path, filename)
            images[band_name] = load_image(file_path)
    return images

def prepare_dataset_to_hdf5(root_dir, hdf5_path):
    """ Store images and labels in an HDF5 file for efficient memory usage. """
    mlb = MultiLabelBinarizer()
    labels_list = []

    # Collect all labels to fit the MultiLabelBinarizer
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            label_path = os.path.join(folder_path, f"{folder_name}_labels_metadata.json")
            labels = parse_json_labels(label_path)
            labels_list.append(labels)

    mlb.fit(labels_list)

    with h5py.File(hdf5_path, 'w') as f:
        images_group = f.create_group('images')
        labels_group = f.create_group('labels')

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                images = load_images_from_folder(folder_path)
                label_path = os.path.join(folder_path, f"{folder_name}_labels_metadata.json")
                labels = parse_json_labels(label_path)
                encoded_labels = mlb.transform([labels])[0]

                folder_group = images_group.create_group(folder_name)
                for band_name, image_data in images.items():
                    folder_group.create_dataset(band_name, data=image_data)
                labels_group.create_dataset(folder_name, data=encoded_labels.astype(np.float32))  # Store as float32

hdf5_train = 'data/train_dataset.hdf5'
hdf5_val = 'data/val_dataset.hdf5'
prepare_dataset_to_hdf5('train', hdf5_train)
prepare_dataset_to_hdf5('validation', hdf5_val)
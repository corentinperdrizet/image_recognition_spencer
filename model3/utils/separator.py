""" Script to split the data into training and validation sets. """

import os
import shutil
import numpy as np

def split_data(base_path, train_size=0.8):
    """
    Split the data into training and validation sets.

    Parameters:
    - base_path (str): The base path of the dataset.
    - train_size (float): The proportion of data to be used for training. Default is 0.8.

    Returns:
    - None
    """
    # Paths to source data folders
    source_images = os.path.join(base_path, 'whole_dataset', 'images')
    source_masks = os.path.join(base_path, 'whole_dataset', 'masks')
    
    # Paths to target data folders
    train_images_path = os.path.join(base_path, 'train', 'images')
    train_masks_path = os.path.join(base_path, 'train', 'masks')
    val_images_path = os.path.join(base_path, 'val', 'images')
    val_masks_path = os.path.join(base_path, 'val', 'masks')
    
    # Creation of the folders if they don't already exist
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_masks_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_masks_path, exist_ok=True)
    
    # Files list
    images = [f for f in os.listdir(source_images) if f.endswith('.jpg')]
    masks = [f for f in os.listdir(source_masks) if f.endswith('.png')]
    
    # Ensure images and masks match and are sorted
    images.sort()
    masks.sort()
    
    # Shuffle images with a fixed seed for reproducibility
    indices = np.arange(len(images))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    # Calculate the split index
    split_idx = int(len(images) * train_size)
    
    # Split data into train and val sets
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Function to copy files
    def copy_files(indices, source_folder, target_folder):
        for idx in indices:
            file_name = images[idx]
            shutil.copy(os.path.join(source_folder, file_name), os.path.join(target_folder, file_name))
            # Copy the corresponding mask
            mask_name = masks[idx]
            shutil.copy(os.path.join(source_masks, mask_name), os.path.join(target_folder.replace('images', 'masks'), mask_name))
    
    # Copy files to train and val folders
    copy_files(train_indices, source_images, train_images_path)
    copy_files(val_indices, source_images, val_images_path)

# Use the function
base_path = 'base/path/to/dataset'
split_data(base_path)

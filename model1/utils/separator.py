#################################################
## This script is used to separate the dataset ##
##       into train and validation sets.       ##
#################################################

# Need to have a 'dataset/' folder in 'data/'

import os
import numpy as np
import shutil

dataset_dir = './data/dataset'

train_dir = './data/train'
test_dir = './data/test'

train_ratio = 0.7

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# Parcour each class in the dataset
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        # Create subfolders in train and test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(train_class_dir):
            os.mkdir(train_class_dir)
        if not os.path.exists(test_class_dir):
            os.mkdir(test_class_dir)

        # List of images in the class folder
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # Randomly shuffle the list of images
        np.random.shuffle(images)

        # Separation point for train/test split
        split_idx = int(len(images) * train_ratio)

        # Images for training
        train_images = images[:split_idx]

        # Images for testing
        test_images = images[split_idx:]

        # Copy images to train and test folders
        for image in train_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(train_class_dir, image))
        for image in test_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(test_class_dir, image))

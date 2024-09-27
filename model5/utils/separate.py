import os
import shutil
import numpy as np

# Path to the folder containing the data
data_dir = 'entire_dataset'

# Destination folders
train_dir = os.path.join('train')
val_dir = os.path.join('validation')

# Create the folders if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get a list of all subfolders (each folder is a photo)
all_folders = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

# Shuffle the folders to randomize
np.random.shuffle(all_folders)

# Proportion for the training set
train_ratio = 0.8
train_size = int(train_ratio * len(all_folders))

# Split into training and validation sets
train_folders = all_folders[:train_size]
val_folders = all_folders[train_size:]

# Function to move the folders
def move_folders(folders, destination):
    for folder in folders:
        shutil.move(folder, destination)

# Move the folders to the train and validation directories
move_folders(train_folders, train_dir)
move_folders(val_folders, val_dir)

print(f"Train folders moved: {len(train_folders)}")
print(f"Validation folders moved: {len(val_folders)}")


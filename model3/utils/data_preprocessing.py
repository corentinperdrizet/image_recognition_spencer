import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array


color_to_class = {
    (60, 16, 152): 0,  # Building
    (132, 41, 246): 1,  # Land
    (110, 193, 228): 2,  # Road
    (254, 221, 58): 3,  # Vegetation
    (226, 169, 41): 4,  # Water
    (155, 155, 155): 5  # Unlabeled
}

def convert_to_class_mask(mask_array, tolerance=10):
    """Convert color mask image to class mask with color tolerance."""
    class_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
    for color, class_id in color_to_class.items():
        # Creating a condition that tolerates a small color difference
        lower_bound = np.maximum([0, 0, 0], np.array(color) - tolerance)
        upper_bound = np.minimum([255, 255, 255], np.array(color) + tolerance)
        matches = np.all((mask_array >= lower_bound) & (mask_array <= upper_bound), axis=-1)
        class_mask[matches] = class_id
    return class_mask


def process_image(path):
    """Load image and convert to array."""
    img = load_img(path)
    return img_to_array(img)

class SatelliteDataGenerator(Sequence):
    """Generates data for training and validation."""
    def __init__(self, image_paths, mask_paths, batch_size):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_x_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_paths = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([process_image(x) for x in batch_x_paths])
        batch_y = np.array([convert_to_class_mask(process_image(y)) for y in batch_y_paths])
        return batch_x, batch_y

def get_image_mask_paths(images_dir, masks_dir):
    image_paths = [os.path.join(images_dir, fname) for fname in sorted(os.listdir(images_dir))]
    mask_paths = [os.path.join(masks_dir, fname) for fname in sorted(os.listdir(masks_dir))]
    return image_paths, mask_paths

train_image_paths, train_mask_paths = get_image_mask_paths('data/train/images', 'data/train/masks')
val_image_paths, val_mask_paths = get_image_mask_paths('data/val/images', 'data/val/masks')

# Setup data generators
batch_size = 4
train_gen = SatelliteDataGenerator(train_image_paths, train_mask_paths, batch_size)
val_gen = SatelliteDataGenerator(val_image_paths, val_mask_paths, batch_size)

############################################
####           Preparing data           ####
############################################

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to safely load an image and display an error message in case of an issue
def safe_load_image(image_path, target_size=(350, 350)):
    try:
        # Load and verify the image
        img = Image.open(image_path)
        img = img.resize(target_size)  # Resize the image
        img.load()  # Load the image to ensure it's complete
        return img
    except (IOError, OSError) as e:
        # Display an error message with the path of the corrupted image
        print(f"Error loading image {image_path}: {e}")
        return None

# Creating a custom DataGenerator to use the safe loading function
class CustomImageDataGenerator(ImageDataGenerator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.target_size + (3,), dtype=self.dtype)
        
        for i, j in enumerate(index_array):
            fname = self.filepaths[j]
            try:
                img = safe_load_image(fname, target_size=self.target_size)
                if img is None:
                    print(f"Corrupted image skipped: {fname}")
                    continue  # Skip the corrupted image

                # Convert to array
                try:
                    x = self.img_to_array(img)
                    x = self.standardize(x)
                    batch_x[i] = x
                except Exception as e:
                    print(f"Error converting image {fname}: {e}")
                    continue  # Skip the problematic image
                
            except Exception as e:
                # Add an error message with the image path
                print(f"Error with image {fname}: {e}")
                continue  # Skip the problematic image

        return batch_x

def get_data_generators(train_dir, test_dir):
    # Create generators with CustomImageDataGenerator
    train_datagen = CustomImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                             height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    test_datagen = CustomImageDataGenerator(rescale=1./255)

    # Load data with the custom generator
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(350, 350), batch_size=8, class_mode='binary')
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(350, 350), batch_size=8, class_mode='binary')
    
    return train_generator, test_generator

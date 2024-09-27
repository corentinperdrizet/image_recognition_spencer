# predict.py
# Description: This script predicts and displays the mask of a satellite image, and calculates the proportion of each class.
#
# Usage:
# $ python3 predict.py path/to/an/image
# Output: Displays the original image, the predicted mask, and prints the pixel proportions of each class.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import color_to_class, model
from utils.data_preprocessing import SatelliteDataGenerator, get_image_mask_paths

# Define class to color mappings
class_to_color = {v: np.array(k)/255.0 for k, v in color_to_class.items()}
class_names = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled']
patches = [mpatches.Patch(color=class_to_color[i], label=class_names[i]) for i in range(len(class_names))]

# Function to load an image, make a prediction, display results, and calculate pixel proportions
def process_image_and_predict(path):
    """Loads an image, resizes it, makes a prediction, displays the results and calculates pixel proportions."""
    original_img = load_img(path)
    img = load_img(path, target_size=(512, 512))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_mask = np.argmax(predictions, axis=-1)[0]  # Get the predicted class for each pixel

    # Calculate pixel proportions for each class
    total_pixels = predicted_mask.size
    unique, counts = np.unique(predicted_mask, return_counts=True)
    proportions = dict(zip(unique, counts))

    print("Pixel proportions by class:")
    for class_index, pixel_count in proportions.items():
        class_name = class_names[class_index]
        percentage = (pixel_count / total_pixels) * 100
        print(f"{percentage:.2f}% {class_name}")

    # Display the original image and predicted mask
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    predicted_color_mask = np.zeros((*predicted_mask.shape, 3))
    for c in class_to_color:
        predicted_color_mask[predicted_mask == c] = class_to_color[c]
    plt.imshow(predicted_color_mask)
    plt.title('Predicted Mask')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

# Main execution for command line interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py path/to/image")
        sys.exit(1)

    img_path = sys.argv[1]

    if os.path.isfile(img_path):
        process_image_and_predict(img_path)
    else:
        print(f"Error: {img_path} is not a valid file.")
        sys.exit(1)

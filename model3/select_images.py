"""Description: Select images based on user preferences and bandwidth limit.
The script analyzes all images in the directory 'images_provider' and selects images 
based on the percentage of pixels corresponding to the preferred class. The default 
bandwidth limit is fixed at 10 MB. The execution also displays the selected images.

Default preferences: Building

Example usage:
$ python3 select_images.py Water
-------------------
New preferences: Water
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import model

# Define available class names
class_names = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled']

# Image directory
image_dir = 'images_provider'

# Set up argument parser
parser = argparse.ArgumentParser(description='Classify images and select based on pixel percentage for a specified class and bandwidth limit.',
                                 formatter_class=argparse.RawTextHelpFormatter)  # Enable raw text formatting

# Adding the argument for class preference
parser.add_argument('class_name', nargs='?', default='Building', help='Class to calculate percentage for (default: Building)\n'
                                                                     'Example usage: python3 select_images.py Water\n'
                                                                     '-------------------\n'
                                                                     'Default Preferences: Building')

# Parse arguments
args = parser.parse_args()

# Use the specified class preference or default to 'Building'
class_name = args.class_name

# Validate the class preference
if class_name not in class_names:
    print(f"Error: {class_name} is not a valid class. Choose from: {', '.join(class_names)}")
    sys.exit(1)

# Print the current class preference
print(f"Current class preference: {class_name}")

# Function to calculate the percentage of pixels belonging to the specified class in an image
def calculate_class_percentage(img_path, model, class_name):
    """Calculate the percentage of pixels belonging to the specified class in an image."""
    
    # Get the index of the class
    class_index = class_names.index(class_name)
    
    # Load and preprocess the image
    img = load_img(img_path, target_size=(512, 512))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch of 1

    # Predict the mask using the model
    predictions = model.predict(img_array)
    predicted_mask = np.argmax(predictions, axis=-1)[0]  # Get the predicted class for each pixel

    # Calculate pixel proportions (similar to the predict.py script)
    total_pixels = predicted_mask.size
    unique, counts = np.unique(predicted_mask, return_counts=True)
    proportions = dict(zip(unique, counts))

    # Calculate the percentage of the specified class
    class_pixels = proportions.get(class_index, 0)
    class_percentage = (class_pixels / total_pixels) * 100
    return class_percentage

# Function to analyze images and select based on class percentage and bandwidth limit
def select_images(image_dir, model, class_name, bandwidth_limit=10):
    """Analyze images, calculate the class percentage, and select images within the bandwidth limit."""
    
    results = []
    total_size = 0.0

    # Loop through the images in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(image_dir, filename)
            class_percentage = calculate_class_percentage(path, model, class_name)
            file_size = os.path.getsize(path) / (1024 * 1024)  # File size in MB
            
            results.append({
                'image_id': filename,
                'class_percentage': class_percentage,
                'file_size': file_size
            })
    
    # Sort images by class percentage (highest to lowest)
    sorted_results = sorted(results, key=lambda x: x['class_percentage'], reverse=True)
    
    selected_images = []
    total_size = 0.0
    
    # Select images while respecting the bandwidth limit
    for result in sorted_results:
        if total_size + result['file_size'] <= bandwidth_limit:
            selected_images.append(result)
            total_size += result['file_size']
        else:
            break
    
    return selected_images, total_size

# Function to display selected images with the class percentage
def display_images(images_list, image_dir, class_name):
    """Displays the selected images in a grid with the class percentage above each image."""
    
    plt.figure(figsize=(15, 10))  # Set the figure size
    for i, result in enumerate(images_list):
        img_path = os.path.join(image_dir, result['image_id'])
        img = load_img(img_path)  # Load the image
        plt.subplot(len(images_list) // 5 + 1, 5, i + 1)  # Arrange images in a grid
        plt.imshow(img)
        plt.title(f"{result['class_percentage']:.2f}% {class_name}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Define the bandwidth limit
bandwidth_limit = 10  # MB

# Select images based on the class percentage and bandwidth limit
selected_images, total_bandwidth_used = select_images(image_dir, model, class_name, bandwidth_limit)

# Print the list of images to send
print("Images to send (sorted by class percentage):")
for result in selected_images:
    print(result['image_id'])

print(f"Total bandwidth used: {total_bandwidth_used:.2f} MB")

# Display the selected images with the class percentage
display_images(selected_images, image_dir, class_name)

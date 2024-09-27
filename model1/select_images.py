"""Description: Select images based on user preferences and bandwidth limit."""
# Analyze all images in the directory 'images_provider' 
# and select images based on user preferences and bandwidth limit (here fixed at 10 MB).
# The execution also displays the selected images.
#
# Default preferences: AnnualCrop, Forest, Herb.Vegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
#
# Example usage:
# $ python3 select.py Residential Highway Forest
# -------------------
# New preferences: Residential Highway Forest, AnnualCrop, Herb.Vegetation, Industrial, Pasture, PermanentCrop, River, SeaLake

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from keras.preprocessing import image
from config import model, class_labels

# Image directory
image_dir = 'images_provider'

# Set up argument parser
parser = argparse.ArgumentParser(description='Classify images and select based on modified preferences and bandwidth limit.',
                                 formatter_class=argparse.RawTextHelpFormatter)  # Enable raw text formatting

# Adding the argument with a newline character and additional formatting
parser.add_argument('preferences', metavar='P', type=str, nargs='*',
                    help='List of preferences to prioritize. Example: python3 select.py Residential Highway Forest\n'
                         '-------------------\n'  # Visual separator if newline doesn't work
                         'Default Preferences: AnnualCrop, Forest, Herb.Vegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake')

# Parse arguments
args = parser.parse_args()

# Start with user-specified preferences if provided, else use default list
specified_preferences = args.preferences if args.preferences else class_labels

# Ensure only valid classes are specified and maintain order for the rest
user_preferences = specified_preferences + [cls for cls in class_labels if cls not in specified_preferences]

# Print the current effective preferences
print("Current preferences:", user_preferences)

# Map model outputs to user preferences
model_class_indices = {cls: i for i, cls in enumerate(class_labels)}
user_pref_indices = [model_class_indices[cls] for cls in user_preferences if cls in model_class_indices]

# Priority dictionary based on preferences
priority_dict = {key: i for i, key in enumerate(user_preferences)}


# Store results
results = []

# Analyze images
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(image_dir, filename)
        img = image.load_img(path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        predictions = model.predict(img_array)
        # Filter and reorder predictions to match user preferences
        filtered_predictions = [predictions[0][index] for index in user_pref_indices]
        class_label = user_preferences[np.argmax(filtered_predictions)]
        
        results.append({
            'image_id': filename,
            'class': class_label,
            'file_size': os.path.getsize(path) / (1024 * 1024),  # File size in MB
            'priority': priority_dict[class_label]
        })

# Create DataFrame
df = pd.DataFrame(results)

# Sort images by priority and file size
sorted_df = df.sort_values(by=['priority', 'file_size'], ascending=[True, True])

# Select images within bandwidth limit
def select_images(sorted_df, bandwidth_limit=10):
    selected_images = []
    total_size = 0.0
    for _, row in sorted_df.iterrows():
        if total_size + row['file_size'] <= bandwidth_limit:
            selected_images.append(row['image_id'])
            total_size += row['file_size']
        else:
            break
    return selected_images, total_size

# Execute selection
images_to_send, total_bandwidth_used = select_images(sorted_df)
print("Images to send:", images_to_send)
print("Total bandwidth used: {:.2f} MB".format(total_bandwidth_used))

# Display selected images
def display_images(images_list, image_dir):
    plt.figure(figsize=(15, 10))  # Set the figure size
    for i, image_name in enumerate(images_list):
        img_path = os.path.join(image_dir, image_name)
        img = image.load_img(img_path)  # Load the image
        plt.subplot(len(images_list) // 5 + 1, 5, i + 1)  # Arrange images in a grid
        plt.imshow(img)
        plt.title(image_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Call the function to display images
display_images(images_to_send, image_dir)

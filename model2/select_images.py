import os
import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt
from config import model


# Image directory
image_dir = 'images_provider'

# Store results
results = []

# Analyze images
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(image_dir, filename)
        # Load and prepare image for boat model
        img = image.load_img(path, target_size=(512, 512))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image if the model was trained on normalized images
        
        # Predict boat presence
        boat_prediction = model.predict(img_array)[0][0]
        
        # Store results including boat probability
        results.append({
            'image_id': filename,
            'boat_probability': boat_prediction,
            'file_size': os.path.getsize(path) / (1024 * 1024)  # File size in MB
        })

# Create DataFrame
df = pd.DataFrame(results)

# Sort images by boat probability in descending order
sorted_df = df.sort_values(by='boat_probability', ascending=False)

# Function to select images within bandwidth limit, sorted by probability
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
        plt.title(f'Prob: {sorted_df.loc[sorted_df["image_id"] == image_name, "boat_probability"].iloc[0]:.2f}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Call the function to display images
display_images(images_to_send, 'images_provider')

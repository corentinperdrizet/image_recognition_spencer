""" Description: This script is used to predict the class of each 
patch of a big image and display the image with annotations"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from config import model, class_labels

# Define a color palette for different classes
colors = ['gold', 'darkgreen', 'green', 'gray', 'red', 'olive', 'orange', 'pink', 'darkblue', 'turquoise']
# Function to load and prepare the image
def load_and_prepare_image(img_path):
    img = image.load_img(img_path)
    return img

# Function to cut the image into subimages of 256x256
def crop_image_to_patches(img, patch_size=256, stride=500):  # Patch size adjusted to 256x256
    image_patches = []
    img_array = np.array(img)
    for y in range(0, img_array.shape[0] - patch_size + 1, stride):
        for x in range(0, img_array.shape[1] - patch_size + 1, stride):
            patch = img_array[y:y+patch_size, x:x+patch_size]
            image_patches.append((patch, x, y))
    return image_patches

# Function to predict the class of each patch by resizing the patches to 64x64
def predict_patches(patches):
    results = []
    for patch, x, y in patches:
        resized_patch = image.array_to_img(patch).resize((64, 64))  # Resize each patch to 64x64
        img_array = image.img_to_array(resized_patch)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        color = colors[predicted_class_index]  # Assign a color to each class
        results.append((predicted_class, color, x, y))
    return results

# Function to display image with annotations
def display_image_with_annotations(img, predictions):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    for predicted_class, color, x, y in predictions:
        rect_width = 256 
        rect_height = 256
        rect = mpl_patches.Rectangle((x, y), rect_width, rect_height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # Calculate the position of the text
        text_x = x + rect_width / 2
        text_y = y + rect_height - 15
        ax.text(text_x, text_y, predicted_class, color='white', fontsize=6, ha='center', backgroundcolor=color)
    plt.show()

img_path = 'path/to/an/image'
img = load_and_prepare_image(img_path)
patches = crop_image_to_patches(img, patch_size=256, stride=300)
predictions = predict_patches(patches)
display_image_with_annotations(img, predictions)

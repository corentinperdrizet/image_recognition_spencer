""" Displays images and their predicted probabilities for wildfire and
no wildfire. Images are selected randomly from the test dataset directory.  """
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import sys

# Add the project root to the path
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)
project_root = os.path.dirname(parent_directory)
sys.path.append(project_root)

from config import model

def load_and_predict(img_path, model):
    """Loads an image, prepares it and makes a prediction."""
    img = image.load_img(img_path, target_size=(350, 350))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    return predictions[0][0]

def display_images(model, no_wildfire_dir='./data/test/nowildfire', wildfire_dir='./data/test/wildfire', num_images=10):
    """Shows images and predicted probabilities for wildfire and no wildfire."""
    
    # Random selection of images
    wildfire_images = random.sample(os.listdir(wildfire_dir), num_images)
    no_wildfire_images = random.sample(os.listdir(no_wildfire_dir), num_images)
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16)) 
    plt.subplots_adjust(hspace=1)  
    axes = axes.flatten()
    
    # Displaying “no wildfire” images
    for i, img_name in enumerate(no_wildfire_images):
        img_path = os.path.join(no_wildfire_dir, img_name)
        prob = load_and_predict(img_path, model)
        img = image.load_img(img_path, target_size=(350, 350))  # Adapter à ta taille d'image
        color = 'green' if prob < 0.50 else 'red'
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Wildfire probability: {prob:.2f}', color=color, y=0.98)
    
    # Display "wildfire" images
    for i, img_name in enumerate(wildfire_images):
        img_path = os.path.join(wildfire_dir, img_name)
        prob = load_and_predict(img_path, model)
        img = image.load_img(img_path, target_size=(350, 350))
        color = 'red' if prob < 0.50 else 'green'
        axes[i + num_images].imshow(img)
        axes[i + num_images].axis('off')
        axes[i + num_images].set_title(f'Wildfire probability: {prob:.2f}', color=color, y=0.98)
    
    fig.text(0.03, 0.75, '- - - - - - - - No Wildfire - - - - - - - -', ha='center', va='center', rotation='vertical', fontsize=12)
    fig.text(0.03, 0.25, '- - - - - - - - Wildfires - - - - - - - -', ha='center', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout()
    plt.show()

display_images(model)

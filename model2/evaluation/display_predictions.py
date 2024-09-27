""" Displays images and their predicted probabilities for boats and non-boats. 
Images are selected randomly from the test dataset directory.  """
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
    img = image.load_img(img_path, target_size=(512, 512))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    return predictions[0][0]

def display_images(model, not_boat_dir='./data/test/a_not_boat', boat_dir='./data/test/boat', num_images=10):
    """Shows images and predicted probabilities for boats and non-boats."""
    
    # Randomly select images
    boat_images = random.sample(os.listdir(boat_dir), num_images)
    not_boat_images = random.sample(os.listdir(not_boat_dir), num_images)
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16)) 
    plt.subplots_adjust(hspace=1)  
    axes = axes.flatten()
    
    for i, img_name in enumerate(not_boat_images):
        img_path = os.path.join(not_boat_dir, img_name)
        prob = load_and_predict(img_path, model)
        img = image.load_img(img_path, target_size=(512, 512))
        color = 'green' if prob < 0.50 else 'red'
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Boat probability: {prob:.2f}', color=color, y=0.98)
    
    for i, img_name in enumerate(boat_images):
        img_path = os.path.join(boat_dir, img_name)
        prob = load_and_predict(img_path, model)
        img = image.load_img(img_path, target_size=(512, 512))
        color = 'red' if prob < 0.50 else 'green'
        axes[i + num_images].imshow(img)
        axes[i + num_images].axis('off')
        axes[i + num_images].set_title(f'Boat probability: {prob:.2f}', color=color, y=0.98)
    
    # Adding side labels
    fig.text(0.03, 0.75, '- - - - - - - - No Boat - - - - - - - -', ha='center', va='center', rotation='vertical', fontsize=12)
    fig.text(0.03, 0.25, '- - - - - - - - Boats - - - - - - - -', ha='center', va='center', rotation='vertical', fontsize=12)
    
    plt.subplots_adjust(left=0.15)

    plt.tight_layout()
    plt.show()

display_images(model)


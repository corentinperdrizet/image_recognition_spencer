# predict.py
# Description: This file predicts whether an image contains a boat or not, provided as a command-line argument.
#
# Usage:
# $ python3 predict.py path/to/an/image
# Output:
# Boat probability: 0.85

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from config import model

# Function to load and predict the class (boat or not) of an image
def predict_boat(img_path, model):
    img = image.load_img(img_path, target_size=(512, 512))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch of 1
    img_array /= 255.0  # Rescale as during training

    predictions = model.predict(img_array)
    return predictions[0][0]  # Return the probability of the image containing a boat

# Main execution for command line interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py path/to/image")
        sys.exit(1)

    img_path = sys.argv[1]

    if os.path.isfile(img_path):
        prob = predict_boat(img_path, model)
        print(f"Boat probability: {prob:.2f}")
    else:
        print(f"Error: {img_path} is not a valid file.")
        sys.exit(1)

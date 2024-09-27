# predict.py
# Description: This file predicts the class of an image provided as a command-line argument.
#
# Usage:
# $ python3 predict.py path/to/an/image
# Output:
# Prediction: Forest

import os
import sys
import numpy as np
from tensorflow.keras.preprocessing import image
from config import model, class_labels

# Function to predict the class of an image
def predict_image(img_path, model, class_labels):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch of 1
    img_array /= 255.0  # Rescale as during training

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    
    return predicted_class

# Main execution for command line interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py path/to/image")
        sys.exit(1)

    img_path = sys.argv[1]

    if os.path.isfile(img_path):
        predicted_class = predict_image(img_path, model, class_labels)
        print(f"Prediction: {predicted_class}")
    else:
        print(f"Error: {img_path} is not a valid file.")
        sys.exit(1)

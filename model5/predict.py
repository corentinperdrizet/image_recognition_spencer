import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
from config import model, mlb

# Load an RGB image from the specific bands (for visualization or display purposes)
def load_rgb_image(folder_path):
    bands = ['B04', 'B03', 'B02']
    images = []
    for band in bands:
        file_path = os.path.join(folder_path, f"{folder_path.split('/')[-1]}_{band}.tif")
        with Image.open(file_path) as img:
            images.append(np.array(img))

    image_rgb = np.dstack(images)
    image_rgb = image_rgb / image_rgb.max()
    return image_rgb

# Load the individual bands as arrays
def load_image(file_path):
    with Image.open(file_path) as img:
        image_array = np.array(img)
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=-1)
        image_tensor /= 255.0  # Normalize
    return image_tensor.numpy()

# Load all bands from a folder for prediction
def load_images_from_folder(folder_path):
    band_files = {'B01': None, 'B02': None, 'B03': None, 'B04': None, 'B05': None, 'B06': None,
                  'B07': None, 'B08': None, 'B09': None, 'B11': None, 'B12': None, 'B8A': None}
    
    for filename in os.listdir(folder_path):
        for band in band_files.keys():
            if band in filename:
                band_files[band] = load_image(os.path.join(folder_path, filename))
    return band_files

# Prepare inputs for the model
def prepare_inputs(images):
    img_20 = np.stack([images['B01'], images['B09']], axis=-1)  # Stack the 20m bands
    img_60 = np.stack([images['B05'], images['B06'], images['B07'], images['B8A'], images['B11'], images['B12']], axis=-1)  # Stack the 60m bands
    img_120 = np.stack([images['B02'], images['B03'], images['B04'], images['B08']], axis=-1)  # Stack the 120m bands

    input_20 = np.expand_dims(np.squeeze(img_20), axis=0)
    input_60 = np.expand_dims(np.squeeze(img_60), axis=0)
    input_120 = np.expand_dims(np.squeeze(img_120), axis=0)

    return [input_20, input_60, input_120]

# Predict and print the classes for an image folder
def predict_image(model, folder_path, mlb):
    # Load bands and prepare inputs
    images = load_images_from_folder(folder_path)
    inputs = prepare_inputs(images)
    
    # Make predictions
    predictions = model.predict(inputs)
    predicted_labels = (predictions > 0.5).astype(int)
    
    # Extract predicted class names
    predicted_classes = [mlb.classes_[i] for i, val in enumerate(predicted_labels[0]) if val == 1]
    
    # Print the predicted classes
    print(f"Predictions for {folder_path.split('/')[-1]}: {', '.join(predicted_classes) if predicted_classes else 'No classes predicted'}")

# Main
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py path/to/image_folder")
        sys.exit(1)

    # Path to the image folder
    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)
    
    # Run prediction
    predict_image(model, folder_path, mlb)

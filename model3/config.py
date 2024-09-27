# Description: Configuration file for the project

from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/model3.h5')

# Define class and color mapping
color_to_class = {
    (60, 16, 152): 0,  # Building
    (132, 41, 246): 1,  # Land
    (110, 193, 228): 2,  # Road
    (254, 221, 58): 3,  # Vegetation
    (226, 169, 41): 4,  # Water
    (155, 155, 155): 5  # Unlabeled
}

# Path to the dataset
train_dir = './data/train'
test_dir = './data/val'
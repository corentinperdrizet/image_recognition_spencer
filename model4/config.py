# Description: Configuration file for the project

from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/model4.h5')

# Path to the dataset
train_dir = './data/train'
test_dir = './data/test'
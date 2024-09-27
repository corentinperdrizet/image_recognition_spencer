# Description: Configuration file for the project

from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/model1.h5')

# Define class
class_labels = ['AnnualCrop', 'Forest', 'Herb.Vegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Path to the dataset
train_dir = './data/train'
test_dir = './data/test'
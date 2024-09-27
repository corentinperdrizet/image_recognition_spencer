# Description: Configuration file for the project
from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/model2.h5')
 
train_dir = './data/train'
test_dir = './data/test'
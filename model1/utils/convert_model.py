# Description: This script converts the model from a keras model
# to a h5 model without the optimizer (saving only the weights).
from tensorflow.keras.models import load_model
from model import build_model

# Loading the model
model_path = 'path/to/model1.keras'
model = load_model(model_path)

# Save the weights of the model
weights_path = 'same/path/Tmp.weights.h5' # Create a new file that you can delete after
model.save_weights(weights_path)

new_model = build_model()
new_model.load_weights(weights_path)

# Save the model without the optimizer
new_model.save('path/to/model1.h5', include_optimizer=False)

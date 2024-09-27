# Description: This script converts the model from a keras model
# to a h5 model without the optimizer (saving only the weights).
from tensorflow.keras.models import load_model
from model import build_model

# Loading the model
model_path = 'models/model5.keras'
model = load_model(model_path)

# Save the weights of the model
weights_path = 'models/Tmp.weights.h5'
model.save_weights(weights_path)

new_model = build_model()
new_model.load_weights(weights_path)

# Save the model without the optimizer
new_model.save('models/model5.h5', include_optimizer=False)

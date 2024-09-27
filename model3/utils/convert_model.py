# Description: This script converts the model from a keras model
# to a h5 model without the optimizer (saving only the weights).
from tensorflow.keras.models import load_model
from model_unet import build_unet
from model import build_Smaller_unet
from XSmall_Unet import build_XSmall_Unet
from AnotherModel import build_another_unet
from SmallestModel import build_smallest_unet

# Loading the model
model_path = 'models/model3.keras'
model = load_model(model_path)

# Save the weights of the model
weights_path = 'models/Tmp.weights.h5'
model.save_weights(weights_path)

new_model = build_smallest_unet((512, 512, 3))
new_model.load_weights(weights_path)

# Save the model without the optimizer
new_model.save('models/model3.h5', include_optimizer=False)

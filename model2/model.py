############################################
####        Building the model          ####
############################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_model():

    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001), input_shape=(512, 512, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        GlobalMaxPooling2D(),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    return model

# Summary of the model
model = build_model()
model.summary()


# Uncomment bellow for a visual representation of the model

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# Uncomment bellow for visualizing the layers of the model

# import visualkeras
# visualkeras.layered_view(model).show()
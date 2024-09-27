from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

# F1-Score custom metric, another metric than the accuracy made for multi-label classification
def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    precision = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0) / (tf.reduce_sum(tf.cast(y_pred, 'float32'), axis=0) + tf.keras.backend.epsilon())
    recall = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0) / (tf.reduce_sum(tf.cast(y_true, 'float32'), axis=0) + tf.keras.backend.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

# F1-Score custom metric, another metric than the accuracy made for multi-label classification
def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    precision = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0) / (tf.reduce_sum(tf.cast(y_pred, 'float32'), axis=0) + tf.keras.backend.epsilon())
    recall = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0) / (tf.reduce_sum(tf.cast(y_true, 'float32'), axis=0) + tf.keras.backend.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)

def build_model():
    # Input
    input_20 = Input(shape=(20, 20, 2))  # 2 bands for B01, B09
    input_60 = Input(shape=(60, 60, 6))  # 6 bands for B05, B06, B07, B8A, B11, B12
    input_120 = Input(shape=(120, 120, 4))  # 4 bands for B02, B03, B04, B08

    # Processing branches for each channel group
    def conv_branch20(inputs):
        num_filters = 64
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = GlobalMaxPooling2D()(x)
        x = Flatten()(x)
        return x

    def conv_branch60(inputs):
        num_filters = 8
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = GlobalMaxPooling2D()(x)
        x = Flatten()(x)
        return x

    def conv_branch120(inputs):
        num_filters = 8
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = GlobalMaxPooling2D()(x)

        x = Flatten()(x)
        return x

    # Apply the convolution branches
    branch_20 = conv_branch20(input_20)
    branch_60 = conv_branch60(input_60)
    branch_120 = conv_branch120(input_120)

    # Merging outputs
    merged = Concatenate()([branch_20, branch_60, branch_120])

    # Couches fully connected
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(merged)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    output = Dense(43, activation='sigmoid')(x) # 43 Class

    # Building and compilation of the model
    model = Model(inputs=[input_20, input_60, input_120], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall(), f1_score])

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
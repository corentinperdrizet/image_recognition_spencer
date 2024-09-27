############################################
####         Training the model         ####
############################################
import tensorflow as tf
import os
import shutil
import datetime
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from model import build_unet
from utils.data_preprocessing import SatelliteDataGenerator, get_image_mask_paths


# Clear logs
log_dir_root = "logs/fit/"
if os.path.exists(log_dir_root):
    shutil.rmtree(log_dir_root)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Callbacks
callbacks = [
    ModelCheckpoint(
        'models/SmallestModel.keras', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    ),
    TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.1, 
        patience=4, 
        min_lr=0.000001, 
        verbose=1
    )
]

unet_model = build_unet((512, 512, 3))


train_image_paths, train_mask_paths = get_image_mask_paths('data/train/images', 'data/train/masks')
val_image_paths, val_mask_paths = get_image_mask_paths('data/val/images', 'data/val/masks')

# Setup data generators
batch_size = 4
train_gen = SatelliteDataGenerator(train_image_paths, train_mask_paths, batch_size)
val_gen = SatelliteDataGenerator(val_image_paths, val_mask_paths, batch_size)

# Model compilation
unet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Model training
history = unet_model.fit(
    train_gen, 
    epochs=100, 
    validation_data=val_gen,
    callbacks=callbacks
)

############################################
####         Training the model         ####
############################################

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model
import datetime
import os
from model import build_model
from utils.data_preprocessing import get_data_generators
from config import test_dir, train_dir
import shutil


def get_callbacks():

    # Clear logs
    log_dir_root = "logs/fit/"
    if os.path.exists(log_dir_root):
        shutil.rmtree(log_dir_root)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Save the best model
    checkpoint = ModelCheckpoint(
        'models/newTest4.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Reduce learning rate when the model stops improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.1,
        patience=4,
        min_lr=0.00001,
        verbose=1
    )

    return [checkpoint, reduce_lr, tensorboard_callback]

from tensorflow.keras.optimizers import Adam

def compile_model(model, learning_rate):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_generator, test_generator, callbacks):
    history = model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=callbacks)
    return history

# Build and compile the model
model = build_model()
model = compile_model(model, learning_rate=0.001)

train_generator, test_generator = get_data_generators(train_dir, test_dir)

train_model(model, train_generator, test_generator, get_callbacks())


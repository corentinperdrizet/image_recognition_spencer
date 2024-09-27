############################################
####           Preparing data           ####
############################################

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, test_dir):
    # Data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Loading data
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
    
    return train_generator, test_generator

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
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(512, 512), batch_size=16, class_mode='binary')
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(512, 512), batch_size=16, class_mode='binary')
    
    return train_generator, test_generator

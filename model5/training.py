# training.py

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model import build_model 
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from model import f1_score
from sklearn.metrics import f1_score as sklearn_f1_score

def data_generator(images_group, labels_group, batch_size):
    """ Generator function for HDF5 data. """
    keys = list(images_group.keys())
    np.random.shuffle(keys)  # Shuffle keys for each epoch
    while True:
        for i in range(0, len(keys), batch_size):
            batch_images_20, batch_images_60, batch_images_120 = [], [], []
            batch_labels = []
            for key in keys[i:i + batch_size]:
                img_20 = np.concatenate([images_group[key]['B01'][:], images_group[key]['B09'][:]], axis=-1)
                img_60 = np.concatenate([images_group[key]['B05'][:], images_group[key]['B06'][:], images_group[key]['B07'][:], images_group[key]['B8A'][:], images_group[key]['B11'][:], images_group[key]['B12'][:]], axis=-1)
                img_120 = np.concatenate([images_group[key]['B02'][:], images_group[key]['B03'][:], images_group[key]['B04'][:], images_group[key]['B08'][:]], axis=-1)

                batch_images_20.append(img_20)
                batch_images_60.append(img_60)
                batch_images_120.append(img_120)
                batch_labels.append(labels_group[key][:])

            yield ((np.stack(batch_images_20), np.stack(batch_images_60), np.stack(batch_images_120)), np.stack(batch_labels))



def train_model_from_hdf5(hdf5_train_path, hdf5_val_path, model, epochs=100, batch_size=32):
    """ Train a model using data from HDF5 files. """

    model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy', Precision(), Recall(), f1_score])
    
    checkpoint_callback = ModelCheckpoint(
        'models/bestmodel.keras', save_best_only=True, monitor='val_f1_score', mode='max', verbose=1
    )
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.1, patience=5, min_lr=0.00001, verbose=1
    )

    with h5py.File(hdf5_train_path, 'r') as train_f, h5py.File(hdf5_val_path, 'r') as val_f:
        train_images_group = train_f['images']
        train_labels_group = train_f['labels']
        val_images_group = val_f['images']
        val_labels_group = val_f['labels']

        output_signature = (
            (
                tf.TensorSpec(shape=(None, 20, 20, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 60, 60, 6), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 120, 120, 4), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None, 43), dtype=tf.float32)
        )

        train_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(train_images_group, train_labels_group, batch_size),
            output_signature=output_signature
        )

        val_dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(val_images_group, val_labels_group, batch_size),
            output_signature=output_signature
        )


        history = model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=len(list(train_images_group.keys())) // batch_size,
            validation_data=val_dataset,
            validation_steps=len(list(val_images_group.keys())) // batch_size,
            callbacks=[checkpoint_callback, reduce_lr_callback]
        )

    return history

# Example usage
hdf5_train = 'data/train_dataset.hdf5'
hdf5_val = 'data/val_dataset.hdf5'
model = build_model()
train_model_from_hdf5(hdf5_train, hdf5_val, model)
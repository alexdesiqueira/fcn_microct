from tensorflow.keras.callbacks import ModelCheckpoint

import constants as const
import data
import misc
import model
import numpy as np
import os
import tensorflow as tf

# setting network constants.
NETWORK = 'unet'  # available: 'unet', 'tiramisu'
FILENAME = f'larson_{NETWORK}.hdf5'
BATCH_SIZE = 1
TARGET_SIZE = (288, 288)

# image and label folders.
FOLDER_BASE = const.FOLDER_TRAINING_CROP
FOLDER_TRAIN = os.path.join(FOLDER_BASE, 'train')
FOLDER_VALIDATE = os.path.join(FOLDER_BASE, 'validate')
SUBFOLDER_IMAGE = 'image'
SUBFOLDER_LABEL = 'label'

# training and validation images.
TRAINING_IMAGES = 70000
VALIDATION_IMAGES = 29900

EPOCHS = 10
STEPS_PER_EPOCH = int(TRAINING_IMAGES // BATCH_SIZE)
STEPS_VALIDATION = int(VALIDATION_IMAGES // BATCH_SIZE)

# augmentation arguments
FILL_MODE = 'nearest'
FLIP_HORIZONTAL = True
FLIP_VERTICAL = True
RANGE_HEIGHT_SHIFT = 0.05
RANGE_ROTATION = 0.1
RANGE_SHEAR = 0.05
RANGE_WIDTH_SHIFT = 0.05
RANGE_ZOOM = 0.05


# preparing TensorFlow
GPUS = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4',
        '/gpu:5', '/gpu:6', '/gpu:7', '/gpu:8', '/gpu:9']
gpus_avail = len(tf.config.experimental.list_physical_devices("GPU"))
mirrored_strategy = tf.distribute.MirroredStrategy(devices=GPUS[:gpus_avail])

print('# Setting hyperparameters')
DATA_GEN_ARGS = dict(rotation_range=RANGE_ROTATION,
                     width_shift_range=RANGE_WIDTH_SHIFT,
                     height_shift_range=RANGE_HEIGHT_SHIFT,
                     shear_range=RANGE_SHEAR,
                     zoom_range=RANGE_ZOOM,
                     horizontal_flip=FLIP_HORIZONTAL,
                     vertical_flip=FLIP_VERTICAL,
                     fill_mode=FILL_MODE)

train_gen = data.train_generator(batch_size=BATCH_SIZE,
                                 train_path=FOLDER_TRAIN,
                                 image_folder=SUBFOLDER_IMAGE,
                                 label_folder=SUBFOLDER_LABEL,
                                 aug_dict=DATA_GEN_ARGS,
                                 target_size=TARGET_SIZE,
                                 save_to_dir=None)

valid_gen = data.train_generator(batch_size=BATCH_SIZE,
                                 train_path=FOLDER_VALIDATE,
                                 image_folder=SUBFOLDER_IMAGE,
                                 label_folder=SUBFOLDER_LABEL,
                                 aug_dict=DATA_GEN_ARGS,
                                 target_size=TARGET_SIZE,
                                 save_to_dir=None)

print('# Processing')
with mirrored_strategy.scope():
    if NETWORK == 'unet':
        model = model.unet(input_size=(*TARGET_SIZE, 1))
    elif NETWORK == 'tiramisu':
        model = model.tiramisu(input_size=(*TARGET_SIZE, 1))
    else:
        raise ValueError(f'Model {NETWORK} is not available.')

    checkpoint = ModelCheckpoint(filepath=FILENAME,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)

    history = model.fit(train_gen,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        validation_data=valid_gen,
                        validation_steps=STEPS_VALIDATION,
                        verbose=1,
                        callbacks=[checkpoint])

print('# Saving indicators')
misc.save_callbacks_csv(history, filename_base=FILENAME)

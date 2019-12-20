from itertools import product
from model import tiramisu
from tensorflow.keras.callbacks import ModelCheckpoint
from os.path import join, isfile

import data
import numpy as np
import tensorflow as tf

print(f'Num GPUs Available: {len(tf.config.experimental.list_physical_devices("GPU"))}\n')
# print(tf.config.experimental.list_physical_devices('GPU'))
# tf.debugging.set_log_device_placement(True)

# base_folder = '/home/alex/data/larson_2019/data/original_training_sample'
base_folder = '/home/alex/data/larson_2019/data/original_288x288'
mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])

print('# Setting hyperparameters')
batch_size = 1
target_size = (288, 288)  # was : (224, 224)
steps_per_epoch = int(60000 // batch_size)  # a total of 60000 crops
validation_steps = int(25000 // batch_size)  # a total of 25000 crops

data_gen_args = dict(rotation_range=0.1,  # rotation
                     width_shift_range=0.05,  # random shifts
                     height_shift_range=0.05,
                     shear_range=0.05,  # shearing transformations
                     zoom_range=0.05,  # zooming
                     horizontal_flip=True,  # flips
                     vertical_flip=True,
                     #featurewise_center=True,  # feature standardization
                     #featurewise_std_normalization=True,
                     fill_mode='nearest')

train_gene = data.train_generator(batch_size=batch_size,
                                  train_path=join(base_folder,
                                                  'train'),
                                  image_folder='image',
                                  label_folder='label',
                                  aug_dict=data_gen_args,
                                  target_size=target_size,
                                  save_to_dir=None)

valid_gene = data.train_generator(batch_size=batch_size,
                                  train_path=join(base_folder,
                                                  'validate'),
                                  image_folder='image',
                                  label_folder='label',
                                  aug_dict=data_gen_args,
                                  target_size=target_size,
                                  save_to_dir=None)

print('# Processing')
with mirrored_strategy.scope():
    model = tiramisu(input_size=(target_size[0], target_size[1], 1))

    filename = 'larson_tiramisu.hdf5'
    if isfile(filename):
        model.load_weights(filename)

    model_checkpoint = ModelCheckpoint(filename,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

    history_callback = model.fit(train_gene,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=10,
                                 validation_data=valid_gene,
                                 validation_steps=validation_steps,
                                 verbose=1,
                                 callbacks=[model_checkpoint])

print('# Saving indicators')
np.savetxt('callbacks/larson_tiramisu_accuracy.csv',
           np.asarray(history_callback.history['accuracy']),
           delimiter=',')

np.savetxt('callbacks/larson_tiramisu_val_accuracy.csv',
           np.asarray(history_callback.history['val_accuracy']),
           delimiter=',')

np.savetxt('callbacks/larson_tiramisu_loss.csv',
           np.asarray(history_callback.history['loss']),
           delimiter=',')

np.savetxt('callbacks/larson_tiramisu_val_loss.csv',
           np.asarray(history_callback.history['val_loss']),
           delimiter=',')

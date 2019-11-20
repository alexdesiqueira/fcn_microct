from itertools import product
from tensorflow.keras.callbacks import ModelCheckpoint
from model import unet
from os.path import join, isfile

import data
import numpy as np
import tensorflow as tf

print(f'Num GPUs Available: {len(tf.config.experimental.list_physical_devices("GPU"))}\n')
# print(tf.config.experimental.list_physical_devices('GPU'))
# tf.debugging.set_log_device_placement(True)

base_folder = '/home/alex/data/larson_2019/data'
mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])

print('# Setting hyperparameters')
batch_size = 4
target_size = (576, 576)
steps_per_epoch = int(15000 // batch_size)  # a total of 15000 crops
validation_steps = int(6250 // batch_size)  # a total of 6250 crops

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
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

    model = unet(input_size=(576, 576, 1))

    filename = 'larson_unet.hdf5'
    if isfile(filename):
        model.load_weights(filename)

    model_checkpoint = ModelCheckpoint(filename,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

    history_callback = model.fit(train_gene,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=25,
                                 validation_data=valid_gene,
                                 validation_steps=validation_steps,
                                 verbose=1,
                                 callbacks=[model_checkpoint])

print('# Saving indicators')
np.savetxt('callbacks/larson_unet_accuracy.csv',
           np.asarray(history_callback.history['accuracy']),
           delimiter=',')

np.savetxt('callbacks/larson_unet_val_accuracy.csv',
           np.asarray(history_callback.history['val_accuracy']),
           delimiter=',')

np.savetxt('callbacks/larson_unet_loss.csv',
           np.asarray(history_callback.history['loss']),
           delimiter=',')

np.savetxt('callbacks/larson_unet_val_loss.csv',
           np.asarray(history_callback.history['val_loss']),
           delimiter=',')

from itertools import product
from tensorflow.keras.callbacks import ModelCheckpoint
from model import unet
from os.path import join

import data
import numpy as np
import tensorflow as tf

print(f'Num GPUs Available: {len(tf.config.experimental.list_physical_devices("GPU"))}\n')
#print(tf.config.experimental.list_physical_devices('GPU'))
#tf.debugging.set_log_device_placement(True)

base_folder = '/home/alex/data/larson_2019/data'

mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])

with mirrored_strategy.scope():  # processing in 4 GPUs
    # Hyperparameters:
    batch_size = 32  # original: 23, (256x256). Can put a larger size because the system shares it
    target_size = (256, 256)
    steps_per_epoch = int(60000 // batch_size)  #was: 60000
    validation_steps = int(25000 // batch_size)  # was: 25000

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
    model = unet(input_size=(256, 256, 1))

    model_checkpoint = ModelCheckpoint('unet_larson.hdf5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

    model.fit(train_gene,
              steps_per_epoch=steps_per_epoch,
              epochs=1,
              validation_data=valid_gene,
              validation_steps=validation_steps,
              verbose=1,
              callbacks=[model_checkpoint])




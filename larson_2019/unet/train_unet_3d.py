from itertools import product
from tensorflow.keras.callbacks import ModelCheckpoint
from model import unet_3d
from os.path import join, isfile

import data
import other_generator
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io, transform

import numpy as np
import os


def adjust_data(image, labels, multiclass, num_class):
    if multiclass:
        image = image / 255
        if len(labels.shape) == 5:
            labels = labels[:, :, :, :, 0]
        else:
            labels = labels[:, :, :, 0]
        aux_labels = np.zeros(labels.shape + (num_class,))

        for num in range(num_class):
            aux_labels[labels == num, num] = 1

        if multiclass:
            batch, rows, cols, classes = aux_labels.shape
            aux_labels = np.reshape(aux_labels, (batch, rows*cols, classes))
        else:
            rows, cols, classes = aux_labels.shape
            aux_labels = np.reshape(aux_labels, (rows*cols, classes))
        labels = aux_labels

    elif np.max(image) > 1:
        image = image / 255
        labels = labels / 255
        labels[labels > 0.5] = 1
        labels[labels <= 0.5] = 0

    return (image, labels)


def test_generator(test_path, target_size=(32, 32, 32), pad_width=4,
                   multiclass=False, as_gray=True):
    images = io.ImageCollection(os.path.join(test_path, '*.tif'))
    for image in images:
        image = image / 255
        if image.shape != target_size:
            image = transform.resize(image, target_size)
        # padding image to correct slicing after
        image = np.pad(image, pad_width=pad_width, mode='reflect')
        if not multiclass:
            image = np.reshape(image, image.shape+(1,))
        image = np.reshape(image, (1,)+image.shape)
        yield image


def train_generator(batch_size, train_path, image_folder, label_folder,
                    aug_dict, image_color_mode='grayscale',
                    label_color_mode='grayscale', image_save_prefix='image',
                    label_save_prefix='label', multiclass=False,
                    num_class=2, save_to_dir=None, target_size=(40, 40, 40),
                    seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and label_datagen
    to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set
    save_to_dir = "your path"
    '''
    image_datagen = other_generator.ChunkDataGenerator(**aug_dict)
    label_datagen = other_generator.ChunkDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed)

    train_generator = zip(image_generator, label_generator)

    for (image, label) in train_generator:
        image, label = adjust_data(image, label, multiclass, num_class)
        yield (image, label)


print(f'Num GPUs Available: {len(tf.config.experimental.list_physical_devices("GPU"))}\n')
# print(tf.config.experimental.list_physical_devices('GPU'))
# tf.debugging.set_log_device_placement(True)

base_folder = '/home/alex/data/larson_2019/data/original_40x40x40'
mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0'])

print('# Setting hyperparameters')
batch_size = 1
target_size = (40, 40, 40)
steps_per_epoch = int(115200 // batch_size)  # a total of 115200 crops
validation_steps = int(32920 // batch_size)  # a total of 32920 crops

data_gen_args = dict(rotation_range=0.1,  # rotation
                     width_shift_range=0.05,  # random shifts
                     height_shift_range=0.05,
                     shear_range=0.05,  # shearing transformations
                     zoom_range=0.05,  # zooming
                     horizontal_flip=True,  # flips
                     vertical_flip=True,
                     fill_mode='nearest')

train_gene = train_generator(batch_size=batch_size,
                             train_path=join(base_folder,
                                             'train'),
                             image_folder='image',
                             label_folder='label',
                             aug_dict=data_gen_args,
                             target_size=target_size,
                             save_to_dir=None)

valid_gene = train_generator(batch_size=batch_size,
                             train_path=join(base_folder,
                                             'validate'),
                             image_folder='image',
                             label_folder='label',
                             aug_dict=data_gen_args,
                             target_size=target_size,
                             save_to_dir=None)

print('# Processing')
with mirrored_strategy.scope():

    model = unet_3d(input_size=(target_size[0],
                                target_size[1],
                                target_size[2],
                                1)
    )

    filename = 'larson_unet3d.hdf5'

    model_checkpoint = ModelCheckpoint(filename,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

    history_callback = model.fit(train_gene,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=1,
                                 validation_data=valid_gene,
                                 validation_steps=validation_steps,
                                 verbose=1,
                                 callbacks=[model_checkpoint])

print('# Saving indicators')

folder_indicators = 'callbacks'
if not os.path.isdir(folder_indicators):
    os.makedirs(folder_indicators)

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

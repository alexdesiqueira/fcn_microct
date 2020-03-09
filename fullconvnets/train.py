#!/usr/bin/env python

from tensorflow.keras.callbacks import ModelCheckpoint

import argparse
import constants as const
import data
import os
import sys
import tensorflow as tf
import utils


def main():
    """Main function for train.py. Receives arguments and starts train()."""
    # help strings.
    help_description = """Train available neural networks on Larson et al
    samples."""
    help_networks = """convolutional network to be used in the
                       training. Available networks: 'tiramisu',
                       'tiramisu_3d', 'unet', 'unet_3d'"""
    help_tiramisu_model = """when the used network is a tiramisu, the model to
                             be used. Not necessary when using U-Nets.
                             Available models: 'tiramisu-56', 'tiramisu-67'"""
    help_batch_size = """size of the batches used in the training. Default:
                         2"""
    help_epochs = """how many epochs are used in the training. Default: 5"""

    # creating parser and checking arguments.
    parser = argparse.ArgumentParser(description=help_description,
                                     add_help=True)
    parser.add_argument('-n',
                        '--network',
                        type=str,
                        required=True,
                        help=help_networks)
    parser.add_argument('-t',
                        '--tiramisu_model',
                        type=int,
                        required=False,
                        help=help_tiramisu_model)
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        help=help_batch_size)
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        required=False,
                        help=help_epochs)
    arguments = vars(parser.parse_args())

    network, tiramisu_model, batch_size, epochs = list(arguments.values())
    if not batch_size:
        batch_size = 2
    if not epochs:
        epochs = 5

    # checking if a tiramisu network was provided with a tiramisu model.
    no_tiramisu_but_tiramisu_model = (('tiramisu' not in network) and
                                      (tiramisu_model is not None))
    tiramisu_but_no_tiramisu_model = (('tiramisu' in network) and
                                      (tiramisu_model is None))
    if no_tiramisu_but_tiramisu_model or tiramisu_but_no_tiramisu_model:
        parser.print_help()
        sys.exit(0)

    # starting train().
    train(network, tiramisu_model, batch_size, epochs)

    return None


def train(network, tiramisu_model=None, batch_size=2, epochs=5):
    """Train a fully convolutional network to segment the samples
    from Larson et al.

    Parameters
    ----------
    network : str
        Network results to compare with the gold standard.
    tiramisu_model : str or None (default : None)
        Tiramisu model to be used.
    batch_size : int (default : 2)
        Size of the batches used in the training.
    epochs : int (default : 5)
        How many epochs are used in the training.

    Returns
    -------
    None

    Notes
    -----
    network can receive the values in const.AVAILABLE_2D_NETS and
    const.AVAILABLE_3D_NETS.

    tiramisu_model can receive the values 'tiramisu-56' and 'tiramisu-67'.

    This function checks how many GPUs are available and trains the chosen
    network using them all. It also uses augmentation on the input images.
    """
    if tiramisu_model is not None:
        tiramisu_layers = tiramisu_model[8:]
    else:
        tiramisu_layers = ''

    FILENAME = f'larson_{network}{tiramisu_layers}.hdf5'

    train_vars = _training_variables(network)

    STEPS_PER_EPOCH = int(train_vars['training_images'] // batch_size)
    VALIDATION_STEPS = int(train_vars['validation_images'] // batch_size)

    # augmentation arguments.
    FILL_MODE = 'nearest'
    FLIP_HORIZONTAL = True
    FLIP_VERTICAL = True
    RANGE_HEIGHT_SHIFT = 0.05
    RANGE_ROTATION = 0.1
    RANGE_SHEAR = 0.05
    RANGE_WIDTH_SHIFT = 0.05
    RANGE_ZOOM = 0.05

    # preparing TensorFlow to operate in all avaliable GPUs.
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

    train_gen = data.train_generator(batch_size=batch_size,
                                     train_path=train_vars['folder_train'],
                                     subfolders=(const.SUBFOLDER_IMAGE,
                                                 const.SUBFOLDER_LABEL),
                                     augmentation_dict=DATA_GEN_ARGS,
                                     target_size=train_vars['target_size'],
                                     save_to_folder=None)

    valid_gen = data.train_generator(batch_size=batch_size,
                                     train_path=train_vars['folder_validate'],
                                     subfolders=(const.SUBFOLDER_IMAGE,
                                                 const.SUBFOLDER_LABEL),
                                     augmentation_dict=DATA_GEN_ARGS,
                                     target_size=train_vars['target_size'],
                                     save_to_folder=None)

    print('# Processing')
    with mirrored_strategy.scope():
        model = utils.network_models(network,
                                     window_shape=train_vars['target_size'])
        if model is None:
            raise('Model not available.')

        checkpoint = ModelCheckpoint(filepath=FILENAME,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True)

        history = model.fit(train_gen,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=epochs,
                            validation_data=valid_gen,
                            validation_steps=VALIDATION_STEPS,
                            verbose=1,
                            callbacks=[checkpoint])

    print('# Saving indicators')
    utils.save_callbacks_csv(history, filename_base=FILENAME)

    return None


def _training_variables(network):
    """Returns variables to be used in the training."""
    if network in const.AVAILABLE_2D_NETS:
        train_vars = {
            'target_size': const.WINDOW_SHAPE,
            'folder_train': os.path.join(const.FOLDER_TRAINING_CROP, 'train'),
            'folder_validate': os.path.join(const.FOLDER_TRAINING_CROP, 'validate'),
            'training_images': 70000,
            'validation_images': 29800
        }
    elif network in const.AVAILABLE_3D_NETS:
        train_vars = {
            'target_size': const.WINDOW_SHAPE_3D,
            'folder_train': os.path.join(const.FOLDER_TRAINING_CROP_3D, 'train'),
            'folder_validate': os.path.join(const.FOLDER_TRAINING_CROP_3D, 'validate'),
            'training_images': 325844,
            'validation_images': 134832
        }
    return train_vars


if __name__ == '__main__':
    main()

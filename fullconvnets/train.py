#!/usr/bin/env python

from tensorflow.keras.callbacks import ModelCheckpoint
from typing import Dict, Union

import argparse
import constants as const
import data
import json
import numpy as np
import os
import sys
import tensorflow as tf
import utils


def main() -> None:
    """Main function for train.py. Receives arguments and starts train()."""
    # creating parser and checking arguments.
    help_description = """Train available neural networks on Larson et al
    samples."""
    parser = argparse.ArgumentParser(description=help_description,
                                     add_help=True)

    # argument --network.
    help_networks = """convolutional network to be used in the
                       training. Available networks: 'tiramisu',
                       'tiramisu_3d', 'unet', 'unet_3d'"""
    parser.add_argument('-n',
                        '--network',
                        type=str,
                        required=True,
                        help=help_networks)

    # argument --tiramisu_model.
    help_tiramisu_model = """when the used network is a tiramisu, the model to
                             be used. Not necessary when using U-Nets.
                             Available models: 'tiramisu-56', 'tiramisu-67'"""
    parser.add_argument('-t',
                        '--tiramisu_model',
                        type=str,
                        required=False,
                        help=help_tiramisu_model)

    # argument --train_vars.
    help_train_vars = """JSON file containing the training variables 'target_size',
                         'folder_train', 'folder_validate', 'training_images',
                         'validation_images'. Defaults: based on constants.py
                         to train Larson et al samples"""
    parser.add_argument('-v',
                        '--train_vars',
                        type=str,
                        required=False,
                        help=help_train_vars)

    # argument --batch_size.
    help_batch_size = """size of the batches used in the training (optional).
                         Default: 2"""
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        help=help_batch_size)

    # argument --epochs.
    help_epochs = """how many epochs are used in the training (optional).
                     Default: 5"""
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        required=False,
                        help=help_epochs)

    # argument --weights.
    help_weights = """output containing weight coefficients. Default:
                      weights_<NETWORK>.hdf5"""
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=False,
                        help=help_weights)

    arguments = vars(parser.parse_args())

    network, tiramisu_model, train_vars, batch_size, epochs, weights = list(arguments.values())
    # checking if batch_size and epochs are empty.
    if not batch_size:
        batch_size = 2
    if not epochs:
        epochs = 5

    # checking if a tiramisu network was provided with a correct model.
    no_tiramisu_but_tiramisu_model = (('tiramisu' not in network) and
                                      (tiramisu_model in const.AVAILABLE_TIRAMISU_MODELS))
    tiramisu_but_no_tiramisu_model = (('tiramisu' in network) and
                                      (tiramisu_model not in const.AVAILABLE_TIRAMISU_MODELS))
    if no_tiramisu_but_tiramisu_model or tiramisu_but_no_tiramisu_model:
        parser.print_help()
        sys.exit(0)

    # starting train().
    train(network, tiramisu_model, train_vars, batch_size, epochs, weights)

    return None


def train(network: str, tiramisu_model: Union[str, None] = None,
          train_vars: Union[str, None] = None, batch_size: int = 2,
          epochs: int = 5, weights: Union[str, None] = None) -> None:
    """Train a fully convolutional network to segment the samples using
    semantic segmentation.

    Parameters
    ----------
    network : str
        Network results to compare with the gold standard.
    tiramisu_model : str or None (default : None)
        Tiramisu model to be used.
    train_vars : str or None (default : None)
        JSON file containing the training variables 'target_size',
        'folder_train', 'folder_validate', 'training_images',
        'validation_images'. If None, uses the training variables
        defined at constants.py to train Larson et al samples.
    batch_size : int (default : 2)
        Size of the batches used in the training.
    epochs : int (default : 5)
        How many epochs are used in the training.
    weights : str or None (default : None)
        Output containing weight coefficients. If None, saves the coefficients
        in `weights_<network>.hdf5`.

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
    if tiramisu_model is None:
        tiramisu_layers = ''
    else:
        tiramisu_layers = tiramisu_model[8:]

    # checking if weights is empty.
    if weights is None:
        weights = f'weights_{network}{tiramisu_layers}.hdf5'

    # checking if train_vars needs to come from constants.py or the JSON file.
    if train_vars:
        train_vars = _read_training_variables(filename=train_vars)
    else:
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
                                     save_to_dir=None)

    # validation data does not need to be augmented.
    valid_gen = data.train_generator(batch_size=batch_size,
                                     train_path=train_vars['folder_validate'],
                                     subfolders=(const.SUBFOLDER_IMAGE,
                                                 const.SUBFOLDER_LABEL),
                                     augmentation_dict=dict(),
                                     target_size=train_vars['target_size'],
                                     save_to_dir=None)

    print('# Processing')
    with mirrored_strategy.scope():
        model = utils.network_models(network,
                                     window_shape=train_vars['target_size'])
        if model is None:
            raise('Model not available.')

        checkpoint = ModelCheckpoint(filepath=weights,
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
    utils.save_callbacks_csv(history, filename_base=weights)

    return None


def _read_training_variables(filename: str) -> Dict[str, int]:
    """Reads train_vars from a JSON file."""
    with open(filename) as file_json:
        train_vars = json.load(file_json)

    expected_keys = ('target_size',
                     'folder_train',
                     'folder_validate',
                     'training_images',
                     'validation_images')

    for key in expected_keys:
        if (key not in train_vars.keys()) or (not train_vars[key]):
            raise RuntimeError(f'{key} is not defined in {filename}.')

    train_vars['target_size'] = np.array(train_vars['target_size'])
    return train_vars


def _training_variables(network: str) -> Dict[str, int]:
    """Returns variables to be used in the training."""
    if network in const.AVAILABLE_2D_NETS:
        train_vars = {
            'target_size': const.WINDOW_SHAPE,
            'folder_train': os.path.join(const.FOLDER_TRAINING_CROP,
                                         const.SUBFOLDER_TRAIN),
            'folder_validate': os.path.join(const.FOLDER_TRAINING_CROP,
                                            const.SUBFOLDER_VALIDATE),
            'training_images': const.NUMBER_TRAIN_IMAGES,
            'validation_images': const.NUMBER_VAL_IMAGES
        }
    elif network in const.AVAILABLE_3D_NETS:
        train_vars = {
            'target_size': const.WINDOW_SHAPE_3D,
            'folder_train': os.path.join(const.FOLDER_TRAINING_CROP_3D,
                                         const.SUBFOLDER_TRAIN),
            'folder_validate': os.path.join(const.FOLDER_TRAINING_CROP_3D,
                                            const.SUBFOLDER_VALIDATE),
            'training_images': const.NUMBER_TRAIN_IMAGES_3D,
            'validation_images': const.NUMBER_VAL_IMAGES_3D
        }
    return train_vars


if __name__ == '__main__':
    main()

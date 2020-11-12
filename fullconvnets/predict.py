from skimage import io
from typing import Dict

import argparse
import constants as const
import json
import numpy as np
import os
import sys
import utils


def main():
    """Main function for predict.py. Receives arguments and starts predict()."""
    # creating parser and checking arguments.
    help_description = """Predict fibers on Larson et al samples using a
    chosen neural network."""
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

    # argument --predict_vars.
    help_predict_vars = """JSON file containing the variables 'folder', 'path',
                           'has_goldstd', 'path_goldstd',
                           'segmentation_interval', and 'registered_path'.
                           Defaults: based on constants.py to predict on Larson
                           et al samples"""
    parser.add_argument('-v',
                        '--predict_vars',
                        type=str,
                        required=False,
                        help=help_predict_vars)

    # argument --weights.
    help_weights = """file containing weight coefficients to be used on the
                      prediction."""
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help=help_weights)

    arguments = vars(parser.parse_args())

    network, tiramisu_model, predict_vars, weights = list(arguments.values())

    # checking if a tiramisu network was provided with a correct model.
    no_tiramisu_but_tiramisu_model = (('tiramisu' not in network) and
                                      (tiramisu_model in const.AVAILABLE_TIRAMISU_MODELS))
    tiramisu_but_no_tiramisu_model = (('tiramisu' in network) and
                                      (tiramisu_model not in const.AVAILABLE_TIRAMISU_MODELS))
    if no_tiramisu_but_tiramisu_model or tiramisu_but_no_tiramisu_model:
        parser.print_help()
        sys.exit(0)

    # starting predict().
    predict(network, tiramisu_model, predict_vars, weights)

    return None


def predict(network, tiramisu_model=None, predict_vars=None, weights=None):
    """Predict fibers on Larson et al samples using a
    chosen neural network.

    Parameters
    ----------
    network : str
        Network results to compare with the gold standard.
    tiramisu_model : str or None (default : None)
        Tiramisu model to be used.
    predict_vars : str or None (default : None)
        JSON file containing the variables 'folder', 'path', 'has_goldstd',
        'path_goldstd', 'segmentation_interval', and 'registered_path'. If None,
        values are based on Larson et al samples.
    weights : str or None (default : None)
        File containing weight coefficients to be used during prediction.

    Returns
    -------
    None

    Notes
    -----
    predict() calculates predictions on all samples in constants.SAMPLES_BUNCH2
    and saves the results on disk. Be sure you have (lots of) disk space
    available, according to the amount of samples you would like to process.

    Examples
    --------
    >>> weights_unet_3d = '../coefficients/larson2019_unet_3d/larson_unet_3d.hdf5'
    >>> predict(network='unet_3d', weights=weights_unet_3d')
    """
    if predict_vars:
        sample = _read_prediction_variables(predict_vars)
        _predict_on_sample(sample, network, tiramisu_model, weights)
    else:
        samples = const.SAMPLES_BUNCH2
        for sample in samples:
            _predict_on_sample(sample, network, tiramisu_model, weights)

    return None


def _predict_on_sample(sample, network, tiramisu_model, weights):
    """Predicts values on sample, using the chosen network and weights.

    Parameters
    ----------
    network : str
        Network results to compare with the gold standard.
    tiramisu_model : str or None (default : None)
        Tiramisu model to be used.
    predict_vars : str or None (default : None)
        JSON file containing the variables 'folder', 'path', 'has_goldstd',
        'path_goldstd', 'segmentation_interval', and 'registered_path'. If None,
        values are based on Larson et al samples.
    weights : str or None (default : None)
        File containing weight coefficients to be used during prediction.

    Returns
    -------
    None
    """
    print(f"# Now reading sample {sample['path']}.")

    # need to check how to deal with this better.
    try:
        if not sample['file_ext']:
            sample['file_ext'] = const.EXT_SAMPLE
    except KeyError:
        sample['file_ext'] = const.EXT_SAMPLE

    pattern = os.path.join(sample['path'], f"*{sample['file_ext']}")
    data_sample = io.ImageCollection(load_pattern=pattern)

    print('# Processing...')
    folder = os.path.join(utils.prediction_folder(network,
                                                  tiramisu_model),
                          sample['folder'])
    utils.process_sample(folder,
                         data=data_sample,
                         weights=weights,
                         network=network)
    if sample['registered_path']:
        pattern = os.path.join(sample['registered_path'],
                               '*' + const.EXT_REG)
        data_sample = io.ImageCollection(load_pattern=pattern)

        print('# Processing registered sample...')
        folder = os.path.join(utils.prediction_folder(network,
                                                      tiramisu_model),
                              f"{sample['folder']}_REG")
        utils.process_sample(folder,
                             data=data_sample,
                             weights=weights,
                             network=network)
    return None


def _read_prediction_variables(filename: str) -> Dict[str, int]:
    """Reads prediction variables from a JSON file."""
    with open(filename) as file_json:
        pred_vars = json.load(file_json)

    expected_keys = ('folder',
                     'path',
                     'file_ext',
                     'has_goldstd',
                     'path_goldstd',
                     'segmentation_interval',
                     'registered_path')

    for key in expected_keys:
        # if (key not in train_vars.keys()) or (not train_vars[key]):
        if (key not in pred_vars.keys()):
            raise RuntimeError(f'{key} is not defined in {filename}.')

    return pred_vars

if __name__ == '__main__':
    main()

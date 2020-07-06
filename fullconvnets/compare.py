#!/usr/bin/env python

import argparse
import constants as const
import os
import sys
import utils


def main():
    """Main function for compare.py. Receives arguments and starts
    compare()."""
    help_description = """Compare segmentation results between a fully
    convolutional network and the gold standard for Larson et al samples."""
    help_networks = """convolutional network to be used in the
                       comparison. Available networks: 'tiramisu',
                       'tiramisu_3d', 'unet', 'unet_3d'"""
    help_tiramisu_model = """when the used network is a tiramisu, the model to
                             be used. Not necessary when using U-Nets.
                             Available models: 'tiramisu-56', 'tiramisu-67'"""

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
                        type=str,
                        required=False,
                        help=help_tiramisu_model)
    arguments = vars(parser.parse_args())

    network, tiramisu_model = list(arguments.values())

    # checking if a tiramisu network was provided with a tiramisu model.
    no_tiramisu_but_tiramisu_model = (('tiramisu' not in network) and
                                      (tiramisu_model is not None))
    tiramisu_but_no_tiramisu_model = (('tiramisu' in network) and
                                      (tiramisu_model is None))
    if no_tiramisu_but_tiramisu_model or tiramisu_but_no_tiramisu_model:
        parser.print_help()
        sys.exit(0)

    # starting compare().
    compare(network, tiramisu_model)

    return None


def compare(network, tiramisu_model=None):
    """Compare segmentation results between a fully convolutional network and
    the gold standard for Larson et al samples.

    Parameters
    ----------
    network : str
        Network results to compare with the gold standard.
    tiramisu_model : str or None (default : None)
        Tiramisu model to be used.

    Returns
    -------
    None

    Notes
    -----
    network can receive the values in const.AVAILABLE_2D_NETS and
    const.AVAILABLE_3D_NETS.

    tiramisu_model can receive the values 'tiramisu-56' and 'tiramisu-67'.
    """
    tiramisu_layers = utils.check_tiramisu_layers(tiramisu_model)
    network_folder = utils.prediction_folder(network=network,
                                             tiramisu_model=tiramisu_model)
    path_coefficients = os.path.join(const.FOLDER_COMP_COEF,
                                     f'{network}{tiramisu_layers}')

    utils.check_path(pathname=path_coefficients)

    for sample in const.SAMPLES_BUNCH2:
        if sample['has_goldstd']:
            print(f"Now processing {sample['folder']}.")
            is_registered = sample['registered_path'] is not None
            data_prediction, data_goldstd = utils.read_data(sample,
                                                            folder_prediction=network_folder,
                                                            is_registered=is_registered)

            data_prediction = data_prediction[slice(*sample['segmentation_interval'])]

            # coefficients will receive the folder name as a filename.
            filename = f"{path_coefficients}/{sample['folder']}-{network}_coefs.csv"
            file_roc = f"{path_coefficients}/{sample['folder']}-{network}_roc_auc.csv"

            _ = utils.measure_all_coefficients(data_prediction,
                                               data_goldstd,
                                               save_coef=True,
                                               filename=filename)
            _ = measure_roc_and_auc(data_prediction, data_goldstd, save_coef=True,
                                    filename=file_roc)
    return None


if __name__ == '__main__':
    main()

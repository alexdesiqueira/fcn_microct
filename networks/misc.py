from skimage import util

import csv
import evaluation
import numpy as np
import os
import utils

CLASS_1 = [128, 128, 128]
CLASS_2 = [128, 0, 0]
CLASS_3 = [192, 192, 128]
CLASS_4 = [128, 64, 128]
CLASS_5 = [60, 40, 222]
CLASS_6 = [128, 128, 0]
CLASS_7 = [192, 128, 128]
CLASS_8 = [64, 64, 128]
CLASS_9 = [64, 0, 128]
CLASS_10 = [0, 0, 0]

COLOR_DICT = np.array([CLASS_1, CLASS_2, CLASS_3, CLASS_4, CLASS_5,
                       CLASS_6, CLASS_7, CLASS_8, CLASS_9, CLASS_10])


def folders_to_process(folder_base, folder_pred, subfolder_pred,
                       folder_gold_std, subfolder_gold_std):
    """Returns lists of folders with correspondents in both prediction and gold
    standard samples.

    Notes
    -----
       The input arguments generated from Larson's gold standard and our
    results are:

    FOLDER_BASE = '~/data/larson_2019/'
    FOLDER_PRED = 'data/<network>/'
    SUBFOLDER_PRED = 'predict'
    FOLDER_GOLD_STD = 'Seg/Bunch2/'
    SUBFOLDER_GOLD_STD = '19_Gray_image'

    Please replace <network> in FOLDER_PRED with 'unet', 'tiramisu', 'unet_3d',
    or 'tiramisu_3d', depending on the predictions you want to use.
    """
    pred_folders, gold_folders = [], []

    base_pred = os.path.join(folder_base, folder_pred)
    base_gold = os.path.join(folder_base, folder_gold_std)

    for folder in [base_pred, base_gold]:
        _assert_folder_exists(folder)

    # checking the path for prediction samples.
    pred_folders = _folder_samples(base_ref=base_pred,
                                   subfolder_ref=subfolder_pred)

    # now, checking the path for Larson's gold standard.
    gold_folders = _folder_samples(base_ref=base_gold,
                                   subfolder_ref=subfolder_gold_std)

    # ensuring the same samples are available in pred_folders and gold_folders
    pred_valid, gold_valid = _suitable_samples(pred_folders, gold_folders)

    return pred_valid, gold_valid


def measure_all_coefficients(data_test, data_gt,
                             calc_coef={'matthews': True, 'dice': True},
                             save_coef=True,
                             filename='coefficients.csv'):
    """Measures all comparison coefficients between two input data.

    Example
    -------
    >>> from skimage import io
    >>> data_bin = io.ImageCollection(load_pattern='res_figures/binary/Test_TIRR_0_1p5_B0p2_*_bin.png',
                                      plugin=None)[1000:2000]
    >>> data_gt = io.ImageCollection(load_pattern='gt_figures/19_Gray_*.tif',
                                     plugin=None)
    >>> all_matthews, all_dice = measure_all_coefficients(data_bin,
                                                          data_gt,
                                                          save_coef=True)
    """
    _assert_same_length(data_test, data_gt)

    all_matthews, all_dice = ['matthews'], ['dice']

    for idx, img_test in enumerate(data_test):
        aux_gt = utils.process_gt_images(data_gt[idx])

        _assert_compatible(img_test, aux_gt)

        aux_confusion = evaluation.confusion_matrix(aux_gt, img_test)
        if calc_coef['matthews']:
            all_matthews.append(evaluation.measure_matthews(aux_confusion))
        if calc_coef['dice']:
            all_dice.append(evaluation.measure_dice(aux_confusion))

    if save_coef:
        if calc_coef['matthews']:
            with open(filename, 'a+') as file_coef:
                coef_writer = csv.writer(file_coef, delimiter=',', newline='')
                coef_writer.writerow(all_matthews)
        if calc_coef['dice']:
            with open(filename, 'a+') as file_coef:
                coef_writer = csv.writer(file_coef, delimiter=',', newline='')
                coef_writer.writerow(all_dice)

    return all_matthews, all_dice


def save_cropped_image(image, index, window_shape=(512, 512), step=512, folder='temp'):
    """Crops image and saves the cropped chunks in disk.

    Parameters
    ----------
    image : ndarray
        Input image.
    index : int
        Reference number to saved files.
    window_shape : integer or tuple of length image.ndim, optional (default : (512, 512))
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length image.ndim, optional (default : 512)
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    folder : str, optional (default : 'temp')
        The folder to save the cropped files.

    Returns
    -------
        None
    """
    img_crop = np.vstack(util.view_as_windows(image,
                                              window_shape=window_shape,
                                              step=step))

    for idx, aux in enumerate(img_crop):
        fname = '%03d_img_crop-%03d.png' % (index, idx)
        io.imsave(os.path.join(folder, fname), aux)
    return None


def save_predictions(save_folder, predictions, multichannel=False,
                     num_class=2):
    for idx, pred in enumerate(predictions):
        if multichannel:
            output = utils.label_visualize(image=pred,
                                           num_class=num_class,
                                           color_dict=COLOR_DICT)
        else:
            output = pred[:, :, 0]
        io.imsave(os.path.join(save_folder, '%03d_predict.png' % (idx)),
                  util.img_as_ubyte(output))
    return None


def _folder_samples(base_ref, subfolder_ref):
    """Returns the folder structure for each sample.
    """
    folders = []
    for base_path, subfolders, _ in os.walk(base_ref):
        if subfolder_ref in subfolders:
            folders.append(os.path.join(base_path, subfolder_ref))
    return folders


def _suitable_samples(pred_folders, gold_folders):
    """Returns the path of each sample contained in both prediction and
    gold standard folders.
    """
    gold_samples = []

    pred_samples = [folder.split('/')[-2] for folder in pred_folders]

    # some folders in Larson's gold standard have a folder named 'Registered',
    # which contains more information that we need for comparison. We need a
    # special condition to be able to get the right path on these folders.
    for folder in gold_folders:
        aux = folder.split('/')
        if 'Registered' in aux:
            gold_samples.append(aux[-3])
        else:
            gold_samples.append(aux[-2])

    valid_samples = list(set(pred_samples) & set(gold_samples))

    pred_idx = [pred_samples.index(sample) for sample in valid_samples]
    gold_idx = [gold_samples.index(sample) for sample in valid_samples]

    pred_valid = [pred_folders[idx] for idx in pred_idx]
    gold_valid = [gold_folders[idx] for idx in gold_idx]

    return pred_valid, gold_valid


def _assert_folder_exists(folder):
    """Raise an error if the folder does not exist."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"The folder '{folder}' does not exist.")
    return None


def _assert_compatible(image_1, image_2):
    """Raise an error if the shape and dtype do not match."""
    if not image_1.shape == image_2.shape:
        raise ValueError('Input images do not have the same dimensions.')
    return None


def _assert_same_length(data_1, data_2):
    """Raise an error if the data does not have the same length."""
    if not len(data_1) == len(data_2):
        raise ValueError('Input data do not have the same length.')
    return None

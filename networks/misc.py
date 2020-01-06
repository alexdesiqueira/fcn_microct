from skimage import util

import csv
import evaluation
import numpy as np
import utils


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
    >>> all_matthews = measure_all_coefficients(data_bin,
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

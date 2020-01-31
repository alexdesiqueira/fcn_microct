from skimage import color, util

import auxiliar
import csv
import numpy as np


def overlap_predictions(image, prediction):
    """Overlaps the prediction on top of the input image.

    Parameters
    ----------
    image : (M, N) array
        Input image.
    prediction : (M, N, P) array
        Predicted results.

    Returns
    -------
    output : (M, N) array
        Overlapped image.

    Notes
    -----
    P in prediction indicates the number of classes.
    """
    if not image.shape == prediction.shape:
        raise

    overlap = util.img_as_ubyte(color.gray2rgb(image))
    overlap[..., 0] = util.img_as_ubyte(prediction > 0.5)

    return overlap


def predict_on_chunk(data, weights, network='unet_3d', n_class=1, pad_width=4,
                     window_shape=(40, 40, 40), step=32):
    """
    """
    model = auxiliar._aux_network(network,
                                  window_shape=window_shape,
                                  n_class=n_class)
    if network is None:
        raise(f'Network {network} not available.')
    model.load_weights(weights)

    data = np.pad(data, pad_width=pad_width)
    chunk_crop = np.vstack(np.hstack(
        util.view_as_windows(data,
                             window_shape=window_shape,
                             step=step)
    ))

    chunk_gen = auxiliar._aux_generator(chunk_crop)

    results = model.predict(chunk_gen, steps=100, verbose=1)  # TODO: need to check the _actual_ amount of steps
    prediction = auxiliar._aux_predict(results)  # TODO: check the alterations needed on this

    return prediction


def predict_on_image(image, weights, network='unet', n_class=1, pad_width=16,
                     window_shape=(288, 288), step=256):
    """
    """
    model = auxiliar._aux_network(network,
                                  window_shape=window_shape,
                                  n_class=n_class)
    if network is None:
        raise(f'Network {network} not available.')
    model.load_weights(weights)

    image = np.pad(image, pad_width=pad_width)
    image_crop = np.vstack(util.view_as_windows(image,
                                                window_shape=window_shape,
                                                step=step))
    image_gen = auxiliar._aux_generator(image_crop)

    results = model.predict(image_gen, steps=100, verbose=1)
    prediction = auxiliar._aux_predict(results)

    return prediction


def process_gt_images(data_gt):
    """
    """
    if np.unique(data_gt).size > 2:
        data_gt = data_gt == 217

    return util.img_as_bool(data_gt)


def read_csv_coefficients(filename):
    """Reads csv coefficients saved in a file."""
    coefs = []
    csv_file = csv.reader(open(filename, 'r'))
    for row in csv_file:
        coefs.append(row)
    return coefs


def regroup_image(image_set, grid_shape=None, pad_width=32,
                  multichannel=False):
    """
    """
    if multichannel:
        image_set = image_set[:,
                              pad_width:-pad_width,
                              pad_width:-pad_width,
                              :]
    else:
        image_set = image_set[:,
                              pad_width:-pad_width,
                              pad_width:-pad_width]

    image = util.montage(image_set,
                         grid_shape=grid_shape,
                         multichannel=multichannel)

    return image

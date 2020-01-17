from model import unet
from skimage import color, util

import csv
import numpy as np

class_1 = [128, 128, 128]
class_2 = [128, 0, 0]
class_3 = [192, 192, 128]
class_4 = [128, 64, 128]
class_5 = [60, 40, 222]
class_6 = [128, 128, 0]
class_7 = [192, 128, 128]
class_8 = [64, 64, 128]
class_9 = [64, 0, 128]
class_0 = [0, 0, 0]

COLOR_DICT = np.array([class_1, class_2, class_3, class_4, class_5,
                       class_6, class_7, class_8, class_9, class_0])


def label_visualize(image, num_class, color_dict):
    if len(image.shape) == 3:
        image = image[:, :, 0]
    output = np.zeros(image.shape + (3,))
    for num in range(num_class):
        output[image == num, :] = color_dict[num]
    return output / 255


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
    overlap[:, :, 0] = util.img_as_ubyte(prediction > 0.5)

    return overlap


def read_csv_coefficients(filename):
    """Reads csv coefficients saved in a file."""
    coefs = []
    csv_file = csv.reader(open(filename, 'r'))
    for row in csv_file:
        coefs.append(row)
    return coefs


def regroup_image(image_set, grid_shape=None, pad_width=32, multichannel=False):
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


def predict_on_image(image, weights, pad_width=16, window_shape=(288, 288), step=256):
    """
    """
    model = unet(input_size=(window_shape[0], window_shape[1], 1))
    model.load_weights(weights)

    image = np.pad(image, pad_width=pad_width)
    image_crop = np.vstack(util.view_as_windows(image,
                                                window_shape=window_shape,
                                                step=step))
    image_gen = _aux_generator(image_crop)

    results = model.predict(image_gen, steps=100, verbose=1)
    prediction = _aux_predict(results)

    return prediction


def process_gt_images(data_gt):
    """
    """
    if np.unique(data_gt).size > 2:
        data_gt = data_gt == 217

    return util.img_as_bool(data_gt)


def _aux_generator(images, multichannel=False):
    """
    """
    for image in images:
        image = image / 255
        if not multichannel:
            image = np.reshape(image, image.shape+(1,))
        image = np.reshape(image, (1,)+image.shape)
        yield image


def _aux_predict(predictions, pad_width=16, grid_shape=(10, 10),
                 num_class=2, multichannel=False):
    """
    """
    depth, rows, cols, colors = predictions.shape

    if multichannel:
        output = np.zeros((depth, rows-2*pad_width, cols-2*pad_width, colors))
        for idx, pred in enumerate(predictions):
            aux_pred = label_visualize(image=pred,
                                       num_class=num_class,
                                       color_dict=COLOR_DICT)
            output[idx] = aux_pred[pad_width:-pad_width,
                                   pad_width:-pad_width,
                                   :]
    else:
        # output = np.zeros((depth, rows-2*pad_width, cols-2*pad_width))
        output = predictions[:,
                             pad_width:-pad_width,
                             pad_width:-pad_width,
                             0]

    output = util.montage(output,
                          grid_shape=grid_shape,
                          multichannel=multichannel)

    return output

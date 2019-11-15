from itertools import product
from skimage import color, io, transform, util

import glob
import numpy as np
import os


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
    rows, cols = image.shape
    overlap = color.gray2rgb(image)

    aux = prediction[..., 0]
    overlap[:, :, 0] = aux*1.5

    return overlap


def regroup_image(crop_images, images_by_side=10, colors=3):

    rows, cols = crop_images[0].shape
    image = np.zeros((rows*images_by_side,
                      cols*images_by_side,
                      colors))

    for idx, (i, j) in enumerate(product(range(images_by_side), repeat=2)):
        image[i*rows: (i+1)*rows, j*cols: (j+1)*cols, colors-1] = crop_images[idx]

    return image


def save_cropped_image(image, index, side=10, folder='temp'):
    old_rows, old_cols = image.shape
    rows, cols = old_rows // side, old_cols // side

    for idx, (i, j) in enumerate(product(range(side), repeat=2)):
        aux_img = image[i*rows:(i+1)*rows, j*cols:(j+1)*cols]
        fname = '%003d_img_crop-%02d.png' % (index, idx)
        io.imsave(os.path.join(folder, fname), aux_img)
    return None


def save_predictions(save_folder, predictions, flag_multi_class=False,
                     num_class=2):
    for idx, pred in enumerate(predictions):
        if flag_multi_class:
            output = label_visualize(image=pred,
                                     num_class=num_class,
                                     color_dict=COLOR_DICT)
        else:
            output = pred[:, :, 0]
        io.imsave(os.path.join(save_folder, '%03d_predict.png' % (idx)),
                  util.img_as_ubyte(output))
    return None

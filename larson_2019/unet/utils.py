from itertools import product
from model import unet, tiramisu
from skimage import color, io, transform, util
from skimage.measure import label
from skimage.segmentation import mark_boundaries

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

    overlap = util.img_as_ubyte(color.gray2rgb(image))
    overlap[:, :, 0] = util.img_as_ubyte(prediction > 0.5)

    return overlap


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

    image = util.montage(image_set, grid_shape=grid_shape, multichannel=multichannel)

    return image


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

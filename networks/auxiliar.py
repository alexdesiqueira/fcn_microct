from models.unet import unet, unet_3d
from models.tiramisu import tiramisu, tiramisu_3d
from skimage import io, util
from tensorflow.keras.backend import clear_session

import constants as const
import numpy as np
import os
import utils


def _aux_generator(images, multichannel=False):
    """
    """
    for image in images:
        image = image / 255
        if not multichannel:
            image = np.reshape(image, image.shape+(1,))
        image = np.reshape(image, (1,)+image.shape)

        yield image


def _aux_imread_prediction(image):
    """
    """
    return io.imread(image) > 127


def _aux_imread_goldstd(image):
    """
    """
    return utils.process_gt_images(io.imread(image))


def _aux_label_visualize(image, color_dict, num_class=2):
    """
    """
    if np.ndim(image) == 3:
        image = image[..., 0]
    output = np.zeros(image.shape + (3,))
    for num in range(num_class):
        output[image == num, :] = color_dict[num]
    return output / 255


def _aux_network(network='unet', window_shape=(288, 288), n_class=1,
                 preset_model='tiramisu-67'):
    """
    """
    available_nets = {}
    if network in ('tiramisu', 'unet'):
        available_nets = {
            'tiramisu': tiramisu(input_size=(*window_shape, n_class),
                                 preset_model=preset_model),
            'unet': unet(input_size=(*window_shape, n_class)),
        }
    elif network in ('tiramisu_3d', 'unet_3d'):
        available_nets = {
            'tiramisu_3d': tiramisu_3d(input_size=(*window_shape, n_class),
                                       preset_model=preset_model),
            'unet_3d': unet_3d(input_size=(*window_shape, n_class)),
        }
    return available_nets.get(network, None)


def _aux_predict(predictions, pad_width=16, grid_shape=(10, 10),
                 num_class=2, multichannel=False):
    """
    """
    depth, rows, cols, colors = predictions.shape
    # helps to crop the interest part of the prediction
    aux_slice = len(grid_shape) * [slice(pad_width, -pad_width)]

    if multichannel:
        output = np.zeros((depth, rows-2*pad_width, cols-2*pad_width, colors))
        for idx, pred in enumerate(predictions):
            aux_pred = _aux_label_visualize(image=pred,
                                            num_class=num_class,
                                            color_dict=const.COLOR_DICT)
            # OLD CODE.
            # output[idx] = aux_pred[pad_width:-pad_width,
            #                        pad_width:-pad_width,
            #                        :]
            # NEW CODE.
            output[idx] = aux_pred[(*aux_slice, slice(None))]
    else:
        # OLD CODE.
        # output = predictions[:,
        #                      pad_width:-pad_width,
        #                      pad_width:-pad_width,
        #                      0]
        # NEW CODE.
        output = predictions[(slice(None), *aux_slice, 0)]

    output = util.montage(output,
                          fill=0,
                          grid_shape=grid_shape,
                          multichannel=multichannel)

    return output


def _aux_prediction_folder(network='unet'):
    """Auxiliar function. Returns the saving folder according to
    the network.
    """
    available_folders = {
        'tiramisu': const.FOLDER_PRED_TIRAMISU,
        'tiramisu_3d': const.FOLDER_PRED_TIRAMISU_3D,
        'unet': const.FOLDER_PRED_UNET,
        'unet_3d': const.FOLDER_PRED_UNET3D,
    }
    return available_folders.get(network, None)


def _aux_process_sample(folder, data, weights, network='unet'):
    """Auxiliar function. Process the sample and overlaps the original
    image with the results.
    """
    FOLDER_PRED = os.path.join(folder, const.SUBFOLDER_PRED)
    FOLDER_OVER = os.path.join(folder, const.SUBFOLDER_OVER)

    for aux in [FOLDER_PRED, FOLDER_OVER]:
        if not os.path.isdir(aux):
            os.makedirs(aux)

    for idx, image in enumerate(data):
        fname = '%06d.png' % (idx)
        # predicting and saving the results.
        prediction = utils.predict_on_image(image,
                                            weights=weights,
                                            network=network)
        io.imsave(os.path.join(FOLDER_PRED, fname),
                  util.img_as_ubyte(prediction))

        # overlapping results on the original image.
        overlap = utils.overlap_predictions(image, prediction)
        io.imsave(os.path.join(FOLDER_OVER, fname),
                  util.img_as_ubyte(overlap))

        clear_session()  # resetting TensorFlow session state
    return None

from skimage import io, util
from sklearn.model_selection import train_test_split

import constants as const
import numpy as np
import os
import utils


FOLDER_TRAIN = 'tests/data_training/cropped_3d/train'
FOLDER_VALIDATE = 'tests/data_training/cropped_3d/validate'
FOLDER_TEST = 'tests/data_training/cropped_3d/test'


def main():
    """
    """
    data_image = io.imread('tests/beadpack_image.tif', plugin=None)
    data_label = util.invert(io.imread('tests/beadpack_label.tif',
                                       plugin=None))

    save_testing_images(data_image, data_label)
    return None


def save_testing_images(data_image, data_label):
    """"""
    # adding necessary padding.
    data_image = np.pad(data_image,
                        pad_width=const.PAD_WIDTH_3D)
    data_label = np.pad(data_label,
                        pad_width=const.PAD_WIDTH_3D)

    # the last 75% of the images will be stored as tests.
    planes = data_image.shape[0]
    test_idx = int(planes * 0.75)

    test_image = data_image[test_idx:]
    test_label = data_label[test_idx:]


    utils.save_cropped_chunk(test_image,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=os.path.join(FOLDER_TEST, 'image'))

    utils.save_cropped_chunk(test_label,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=os.path.join(FOLDER_TEST, 'label'))

    # separating the remainder as training and validation.
    train_image, validate_image, train_label, validate_label = train_test_split(data_image[:test_idx],
                                                                                data_label[:test_idx],
                                                                                test_size=0.4,
                                                                                random_state=1)

    utils.save_cropped_chunk(train_image,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=os.path.join(FOLDER_TRAIN, 'image'))

    utils.save_cropped_chunk(train_label,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=os.path.join(FOLDER_TRAIN, 'label'))


    utils.save_cropped_chunk(validate_image,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=os.path.join(FOLDER_VALIDATE, 'image'))

    utils.save_cropped_chunk(validate_label,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=os.path.join(FOLDER_VALIDATE, 'label'))

    return None


# To train a 3D U-net on the test data, you'd use:
# python train.py -n 'unet_3d' -v 'tests/beadpack_train.json' -o 'beadpack_unet_3d.hdf5'

if __name__ == '__main__':
    main()

from utils import process_gt_images
from skimage import io

import constants as const
import misc
import numpy as np
import os
import shutil


def _imread_goldstd(image):
    return process_gt_images(io.imread(image))


def copy_training_samples():
    """
    """
    # checking if folders exist.
    for folder in [const.FOLDER_TRAIN_IMAGE_ORIG,
                   const.FOLDER_TRAIN_LABEL_ORIG,
                   const.FOLDER_VAL_IMAGE_ORIG,
                   const.FOLDER_VAL_LABEL_ORIG]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    # getting image intervals.
    start, end = const.INTERVAL_TRAIN_CURED
    for number in range(start, end):
        if number < 1000:
            aux_image = 'Reg_0'
            aux_label = '19_Gray_0'
        else:
            aux_image = 'Reg_'
            aux_label = '19_Gray_'
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['registered_path']}/{aux_image}{number}.tif",
                    dst=f"{const.FOLDER_TRAIN_IMAGE_ORIG}")
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['path_goldstd']}/{aux_label}{number}.tif",
                    dst=f"{const.FOLDER_TRAIN_LABEL_ORIG}")

    start, end = const.INTERVAL_VAL_CURED
    for number in range(start, end):
        if number < 1000:
            aux_image = 'Reg_0'
            aux_label = '19_Gray_0'
        else:
            aux_image = 'Reg_'
            aux_label = '19_Gray_'
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['registered_path']}/{aux_image}{number}.tif",
                    dst=f"{const.FOLDER_VAL_IMAGE_ORIG}")
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['path_goldstd']}/{aux_label}{number}.tif",
                    dst=f"{const.FOLDER_VAL_LABEL_ORIG}")

    start, end = const.INTERVAL_TRAIN_WET
    for number in range(start, end):
        if number < 1000:
            aux_image = 'rec_SFRR_2600_B0p2_00'
            aux_label = '19_Gray_0'
        else:
            aux_image = 'rec_SFRR_2600_B0p2_0'
            aux_label = '19_Gray_'
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path']}/{aux_image}{number}.tiff",
                    dst=f"{const.FOLDER_TRAIN_IMAGE_ORIG}")
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path_goldstd']}/{aux_label}{number}.tif",
                    dst=f"{const.FOLDER_TRAIN_LABEL_ORIG}")

    start, end = const.INTERVAL_VAL_WET
    for number in range(start, end):
        if number < 1000:
            aux_image = 'rec_SFRR_2600_B0p2_00'
            aux_label = '19_Gray_0'
        else:
            aux_image = 'rec_SFRR_2600_B0p2_0'
            aux_label = '19_Gray_'
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path']}/{aux_image}{number}.tiff",
                    dst=f"{const.FOLDER_VAL_IMAGE_ORIG}")
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path_goldstd']}/{aux_label}{number}.tif",
                    dst=f"{const.FOLDER_VAL_LABEL_ORIG}")
    return None


def crop_training_samples():
    """
    """
    # checking if folders exist.
    for folder in [const.FOLDER_TRAIN_IMAGE_CROP,
                   const.FOLDER_TRAIN_LABEL_CROP,
                   const.FOLDER_VAL_IMAGE_CROP,
                   const.FOLDER_VAL_LABEL_CROP]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    # reading training images and its labels.
    aux = [os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, '*' + const.EXT_GOLDSTD)]
    data_image = io.ImageCollection(load_pattern=':'.join(aux),
                                    plugin=None)

    aux = [os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, '*' + const.EXT_GOLDSTD)]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=_imread_goldstd,
                                    plugin=None)

    print(f'* Training images: {len(data_image)}; labels: {len(data_label)}')

    for idx, (image, label) in enumerate(zip(data_image, data_label)):
        image = np.pad(image, pad_width=const.PAD_WIDTH)
        label = np.pad(label, pad_width=const.PAD_WIDTH)

        misc.save_cropped_image(image,
                                index=idx,
                                window_shape=const.WINDOW_SHAPE,
                                step=const.STEP,
                                folder=const.FOLDER_TRAIN_IMAGE_CROP)

        misc.save_cropped_image(label,
                                index=idx,
                                window_shape=const.WINDOW_SHAPE,
                                step=const.STEP,
                                folder=const.FOLDER_TRAIN_LABEL_CROP)

    # reading training images and its labels.
    aux = [os.path.join(const.FOLDER_VAL_IMAGE_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_VAL_IMAGE_ORIG, '*' + const.EXT_GOLDSTD)]
    data_image = io.ImageCollection(load_pattern=':'.join(aux),
                                    plugin=None)

    aux = [os.path.join(const.FOLDER_VAL_LABEL_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_VAL_LABEL_ORIG, '*' + const.EXT_GOLDSTD)]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=_imread_goldstd,
                                    plugin=None)

    print(f'* Validation images: {len(data_image)}; labels: {len(data_label)}')

    for idx, (image, label) in enumerate(zip(data_image, data_label)):
        image = np.pad(image, pad_width=const.PAD_WIDTH)
        label = np.pad(label, pad_width=const.PAD_WIDTH)

        misc.save_cropped_image(image,
                                index=idx,
                                window_shape=const.WINDOW_SHAPE,
                                step=const.STEP,
                                folder=const.FOLDER_VAL_IMAGE_CROP)

        misc.save_cropped_image(label,
                                index=idx,
                                window_shape=const.WINDOW_SHAPE,
                                step=const.STEP,
                                folder=const.FOLDER_VAL_LABEL_CROP)
    return None

from utils import process_goldstd_images
from skimage import io, util

import constants as const
import numpy as np
import os
import shutil
import utils


def _imread_goldstd(image):
    return util.img_as_ubyte(process_goldstd_images(io.imread(image)))


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


def crop_training_chunks():
    """
    """
    # checking if folders exist.
    for folder in [const.FOLDER_TRAIN_IMAGE_CROP_3D,
                   const.FOLDER_TRAIN_LABEL_CROP_3D,
                   const.FOLDER_VAL_IMAGE_CROP_3D,
                   const.FOLDER_VAL_LABEL_CROP_3D]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    # reading training images and their labels.
    aux = [os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, '*' + const.EXT_GOLDSTD)]
    data_image = io.ImageCollection(load_pattern=':'.join(aux))
    # we need to concatenate the data to pad the structure before processing it.
    data_image = data_image.concatenate()

    aux = [os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, '*' + const.EXT_GOLDSTD)]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=_imread_goldstd)
    data_label = data_label.concatenate()

    print(f'* Training images: {len(data_image)}; labels: {len(data_label)}')

    data_image = np.pad(data_image, pad_width=const.PAD_WIDTH_3D)
    data_label = np.pad(data_label, pad_width=const.PAD_WIDTH_3D)

    utils.save_cropped_chunk(data_image,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=const.FOLDER_TRAIN_IMAGE_CROP_3D)

    utils.save_cropped_chunk(data_label,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=const.FOLDER_TRAIN_LABEL_CROP_3D)

    # reading validation images and their labels.
    aux = [os.path.join(const.FOLDER_VAL_IMAGE_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_VAL_IMAGE_ORIG, '*' + const.EXT_GOLDSTD)]
    data_image = io.ImageCollection(load_pattern=':'.join(aux))
    data_image = data_image.concatenate()

    aux = [os.path.join(const.FOLDER_VAL_LABEL_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_VAL_LABEL_ORIG, '*' + const.EXT_GOLDSTD)]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=_imread_goldstd)
    data_label = data_label.concatenate()

    print(f'* Validation images: {len(data_image)}; labels: {len(data_label)}')

    data_image = np.pad(data_image, pad_width=const.PAD_WIDTH_3D)
    data_label = np.pad(data_label, pad_width=const.PAD_WIDTH_3D)

    utils.save_cropped_chunk(data_image,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=const.FOLDER_VAL_IMAGE_CROP_3D)

    utils.save_cropped_chunk(data_label,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=const.FOLDER_VAL_LABEL_CROP_3D)
    return None


def crop_training_images():
    """
    """
    # checking if folders exist.
    for folder in [const.FOLDER_TRAIN_IMAGE_CROP,
                   const.FOLDER_TRAIN_LABEL_CROP,
                   const.FOLDER_VAL_IMAGE_CROP,
                   const.FOLDER_VAL_LABEL_CROP]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    # reading training images and their labels.
    aux = [os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, '*' + const.EXT_GOLDSTD)]
    data_image = io.ImageCollection(load_pattern=':'.join(aux))

    aux = [os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, '*' + const.EXT_GOLDSTD)]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=_imread_goldstd)

    print(f'* Training images: {len(data_image)}; labels: {len(data_label)}')

    for idx, (image, label) in enumerate(zip(data_image, data_label)):
        image = np.pad(image, pad_width=const.PAD_WIDTH)
        label = np.pad(label, pad_width=const.PAD_WIDTH)

        utils.save_cropped_image(image,
                                 index=idx,
                                 window_shape=const.WINDOW_SHAPE,
                                 step=const.STEP,
                                 folder=const.FOLDER_TRAIN_IMAGE_CROP)

        utils.save_cropped_image(label,
                                 index=idx,
                                 window_shape=const.WINDOW_SHAPE,
                                 step=const.STEP,
                                 folder=const.FOLDER_TRAIN_LABEL_CROP)

    # reading validation images and their labels.
    aux = [os.path.join(const.FOLDER_VAL_IMAGE_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_VAL_IMAGE_ORIG, '*' + const.EXT_GOLDSTD)]
    data_image = io.ImageCollection(load_pattern=':'.join(aux))

    aux = [os.path.join(const.FOLDER_VAL_LABEL_ORIG, '*' + const.EXT_SAMPLE),
           os.path.join(const.FOLDER_VAL_LABEL_ORIG, '*' + const.EXT_GOLDSTD)]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=_imread_goldstd)

    print(f'* Validation images: {len(data_image)}; labels: {len(data_label)}')

    for idx, (image, label) in enumerate(zip(data_image, data_label)):
        image = np.pad(image, pad_width=const.PAD_WIDTH)
        label = np.pad(label, pad_width=const.PAD_WIDTH)

        utils.save_cropped_image(image,
                                 index=idx,
                                 window_shape=const.WINDOW_SHAPE,
                                 step=const.STEP,
                                 folder=const.FOLDER_VAL_IMAGE_CROP)

        utils.save_cropped_image(label,
                                 index=idx,
                                 window_shape=const.WINDOW_SHAPE,
                                 step=const.STEP,
                                 folder=const.FOLDER_VAL_LABEL_CROP)
    return None

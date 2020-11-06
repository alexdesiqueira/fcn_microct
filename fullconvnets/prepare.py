from utils import process_goldstd_images
from skimage import io, util

import constants as const
import numpy as np
import os
import shutil
import utils


def copy_training_samples():
    """Copies training and validation images to a separated folder.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The folders for training and validation are defined in
    constants.FOLDER_TRAIN_IMAGE_ORIG and constants.FOLDER_VAL_IMAGE_ORIG,
    respectively.

    Example
    -------
    >>> copy_training_samples()
    """
    # checking if folders exist.
    for folder in [const.FOLDER_TRAIN_IMAGE_ORIG,
                   const.FOLDER_TRAIN_LABEL_ORIG,
                   const.FOLDER_VAL_IMAGE_ORIG,
                   const.FOLDER_VAL_LABEL_ORIG]:
        utils.check_path(folder)

    # getting training images; copying them to the train folder.
    start, end = const.INTERVAL_TRAIN_CURED
    for number in range(start, end):
        fname_image, fname_label = _image_filenames(number, sample='cured')
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['registered_path']}/{fname_image}{number}.tif",
                    dst=f"{const.FOLDER_TRAIN_IMAGE_ORIG}")
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['path_goldstd']}/{fname_label}{number}.tif",
                    dst=f"{const.FOLDER_TRAIN_LABEL_ORIG}")

    # getting validation images; copying them to the validation folder.
    start, end = const.INTERVAL_VAL_CURED
    for number in range(start, end):
        fname_image, fname_label = _image_filenames(number, sample='cured')
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['registered_path']}/{fname_image}{number}.tif",
                    dst=f"{const.FOLDER_VAL_IMAGE_ORIG}")
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['path_goldstd']}/{fname_label}{number}.tif",
                    dst=f"{const.FOLDER_VAL_LABEL_ORIG}")

    start, end = const.INTERVAL_TRAIN_WET
    for number in range(start, end):
        fname_image, fname_label = _image_filenames(number, sample='wet')
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path']}/{fname_image}{number}.tiff",
                    dst=f"{const.FOLDER_TRAIN_IMAGE_ORIG}")
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path_goldstd']}/{fname_label}{number}.tif",
                    dst=f"{const.FOLDER_TRAIN_LABEL_ORIG}")

    start, end = const.INTERVAL_VAL_WET
    for number in range(start, end):
        fname_image, fname_label = _image_filenames(number, sample='wet')
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path']}/{fname_image}{number}.tiff",
                    dst=f"{const.FOLDER_VAL_IMAGE_ORIG}")
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path_goldstd']}/{fname_label}{number}.tif",
                    dst=f"{const.FOLDER_VAL_LABEL_ORIG}")
    return None


def crop_training_chunks():
    """Crops training and validation images in chunks.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The folders for training and validation are defined in
    const.FOLDER_TRAIN_IMAGE_CROP_3D and constants.FOLDER_VAL_IMAGE_CROP_3D,
    respectively.

    Chunks are defined with padding and overlapping between them. The
    size of each chunk is defined in constants.WINDOW_SHAPE_3D, the padding is
    given at constants.PAD_WIDTH_3D, and the overlapping is given using smaller
    steps when defining chunks. The step size is defined in constants.STEP_3D.

    Example
    -------
    >>> copy_training_samples()
    >>> crop_training_chunks()
    """
    # checking if folders exist.
    for folder in [const.FOLDER_TRAIN_IMAGE_CROP_3D,
                   const.FOLDER_TRAIN_LABEL_CROP_3D,
                   const.FOLDER_VAL_IMAGE_CROP_3D,
                   const.FOLDER_VAL_LABEL_CROP_3D]:
        utils.check_path(folder)

    # reading training images and their labels.
    aux = [os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, f'*{const.EXT_SAMPLE}'),
           os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, f'*{const.EXT_GOLDSTD}')]
    data_image = io.ImageCollection(load_pattern=':'.join(aux))
    # concatenating the data to pad the structure before processing it.
    data_image = data_image.concatenate()

    aux = [os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, f'*{const.EXT_SAMPLE}'),
           os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, f'*{const.EXT_GOLDSTD}')]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=utils.imread_goldstd)
    data_label = data_label.concatenate()

    print(f'* Training images: {len(data_image)}; labels: {len(data_label)}')
    data_image = np.pad(data_image, pad_width=const.PAD_WIDTH_3D)
    data_label = np.pad(data_label, pad_width=const.PAD_WIDTH_3D)

    print(f'* Window shape: {const.WINDOW_SHAPE_3D}; step: {const.STEP_3D}')
    utils.save_cropped_chunk(data_image,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=const.FOLDER_TRAIN_IMAGE_CROP_3D)

    utils.save_cropped_chunk(util.img_as_ubyte(data_label),
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=const.FOLDER_TRAIN_LABEL_CROP_3D)

    # reading validation images and their labels.
    aux = [os.path.join(const.FOLDER_VAL_IMAGE_ORIG, f'*{const.EXT_SAMPLE}'),
           os.path.join(const.FOLDER_VAL_IMAGE_ORIG, f'*{const.EXT_GOLDSTD}')]
    data_image = io.ImageCollection(load_pattern=':'.join(aux))
    data_image = data_image.concatenate()

    aux = [os.path.join(const.FOLDER_VAL_LABEL_ORIG, f'*{const.EXT_SAMPLE}'),
           os.path.join(const.FOLDER_VAL_LABEL_ORIG, f'*{const.EXT_GOLDSTD}')]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=utils.imread_goldstd)
    data_label = data_label.concatenate()

    print(f'* Validation images: {len(data_image)}; labels: {len(data_label)}')
    data_image = np.pad(data_image, pad_width=const.PAD_WIDTH_3D)
    data_label = np.pad(data_label, pad_width=const.PAD_WIDTH_3D)
    print(f'* Window shape: {const.WINDOW_SHAPE_3D}; step: {const.STEP_3D}')

    utils.save_cropped_chunk(data_image,
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=const.FOLDER_VAL_IMAGE_CROP_3D)

    utils.save_cropped_chunk(util.img_as_ubyte(data_label),
                             window_shape=const.WINDOW_SHAPE_3D,
                             step=const.STEP_3D,
                             folder=const.FOLDER_VAL_LABEL_CROP_3D)
    return None


def crop_training_images():
    """Crops training and validation images in smaller slices.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The folders for training and validation are defined in
    const.FOLDER_TRAIN_IMAGE_CROP and constants.FOLDER_VAL_IMAGE_CROP,
    respectively.

    Chunks are defined with padding and overlapping between them. The
    size of each slice is defined in constants.WINDOW_SHAPE, the padding is
    given at constants.PAD_WIDTH, and the overlapping is given using smaller
    steps when defining slices. The step size is defined in constants.STEP.

    Example
    -------
    >>> copy_training_samples()
    >>> crop_training_chunks()
    """
    # checking if folders exist.
    for folder in [const.FOLDER_TRAIN_IMAGE_CROP,
                   const.FOLDER_TRAIN_LABEL_CROP,
                   const.FOLDER_VAL_IMAGE_CROP,
                   const.FOLDER_VAL_LABEL_CROP]:
        utils.check_path(folder)

    # reading training images and their labels.
    aux = [os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, f'*{const.EXT_SAMPLE}'),
           os.path.join(const.FOLDER_TRAIN_IMAGE_ORIG, f'*{const.EXT_GOLDSTD}')]
    data_image = io.ImageCollection(load_pattern=':'.join(aux))

    aux = [os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, f'*{const.EXT_SAMPLE}'),
           os.path.join(const.FOLDER_TRAIN_LABEL_ORIG, f'*{const.EXT_GOLDSTD}')]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=utils.imread_goldstd)

    print(f'* Training images: {len(data_image)}; labels: {len(data_label)}')

    for idx, (image, label) in enumerate(zip(data_image, data_label)):
        image = np.pad(image, pad_width=const.PAD_WIDTH)
        label = np.pad(label, pad_width=const.PAD_WIDTH)

        utils.save_cropped_image(image,
                                 index=idx,
                                 window_shape=const.WINDOW_SHAPE,
                                 step=const.STEP,
                                 folder=const.FOLDER_TRAIN_IMAGE_CROP)

        utils.save_cropped_image(util.img_as_ubyte(label),
                                 index=idx,
                                 window_shape=const.WINDOW_SHAPE,
                                 step=const.STEP,
                                 folder=const.FOLDER_TRAIN_LABEL_CROP)

    # reading validation images and their labels.
    aux = [os.path.join(const.FOLDER_VAL_IMAGE_ORIG, f'*{const.EXT_SAMPLE}'),
           os.path.join(const.FOLDER_VAL_IMAGE_ORIG, f'*{const.EXT_GOLDSTD}')]
    data_image = io.ImageCollection(load_pattern=':'.join(aux))

    aux = [os.path.join(const.FOLDER_VAL_LABEL_ORIG, f'*{const.EXT_SAMPLE}'),
           os.path.join(const.FOLDER_VAL_LABEL_ORIG, f'*{const.EXT_GOLDSTD}')]
    data_label = io.ImageCollection(load_pattern=':'.join(aux),
                                    load_func=utils.imread_goldstd)

    print(f'* Validation images: {len(data_image)}; labels: {len(data_label)}')

    for idx, (image, label) in enumerate(zip(data_image, data_label)):
        image = np.pad(image, pad_width=const.PAD_WIDTH)
        label = np.pad(label, pad_width=const.PAD_WIDTH)

        utils.save_cropped_image(image,
                                 index=idx,
                                 window_shape=const.WINDOW_SHAPE,
                                 step=const.STEP,
                                 folder=const.FOLDER_VAL_IMAGE_CROP)

        utils.save_cropped_image(util.img_as_ubyte(label),
                                 index=idx,
                                 window_shape=const.WINDOW_SHAPE,
                                 step=const.STEP,
                                 folder=const.FOLDER_VAL_LABEL_CROP)
    return None


def _image_filenames(number, sample='wet'):
    """Returns filenames based on Larson et al images and labels."""
    if sample == 'cured':
        fname_image = 'Reg_'
    elif sample == 'wet':
        fname_image = 'rec_SFRR_2600_B0p2_0'
    fname_label = '19_Gray_'

    if number < 1000:
        fname_image += '0'
        fname_label += '0'

    return fname_image, fname_label

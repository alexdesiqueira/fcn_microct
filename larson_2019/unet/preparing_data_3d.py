import matplotlib.pyplot as plt
import numpy as np
import csv
import statistics as stats
import os
import warnings

from itertools import product
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import binary_fill_holes
from skimage import draw, filters, io, measure, morphology, restoration, util
from skimage.color import label2rgb
from skimage.restoration import denoise_nl_means
from skimage.segmentation import mark_boundaries
from utils import save_cropped_image


folder_nocrop = '/home/alex/data/larson_2019/data_NOCROP/'
folder_save = '/home/alex/data/larson_2019/data/original_40x40_3d/'

train_image_nocrop = os.path.join(folder_nocrop, 'train/image/')
train_label_nocrop = os.path.join(folder_nocrop, 'train/label/')

val_image_nocrop = os.path.join(folder_nocrop, 'validate/image/')
val_label_nocrop = os.path.join(folder_nocrop, 'validate/label/')

train_image_save = folder_save + 'train/image/'
train_label_save = folder_save + 'train/label/'

val_image_save = folder_save + 'validate/image/'
val_label_save = folder_save + 'validate/label/'

for folder in [train_image_save, train_label_save,
               val_image_save, val_label_save]:
    if not os.path.isdir(folder):
        os.makedirs(folder)


def save_cropped_3d(image, window_shape=(32, 32, 32), step=32, folder='temp'):
    """Crops image and saves the cropped chunks in disk.

    Parameters
    ----------
    image : ndarray
        Input image.
    window_shape : integer or tuple of length image.ndim, optional
        (default : (32, 32, 32))
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length image.ndim, optional (default : 32)
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    folder : str, optional (default : 'temp')
        The folder to save the cropped files.

    Returns
    -------
        None
    """
    cube_crop = np.vstack(np.hstack(
        util.view_as_windows(image,
                             window_shape=window_shape,
                             step=step)
    ))

    for idx, aux in enumerate(cube_crop):
        fname = 'cube_crop-%06d.npy' % (idx)
        np.save(os.path.join(folder, fname), aux)
    return None


pad_width = 4
planes, rows, cols = (32, 32, 32)
window_shape = (planes + 2*pad_width,
                rows + 2*pad_width,
                cols + 2*pad_width)
step = 32


# def image_chunks(collection, chunk_size=step+pad_width):
#    """Yield successive n-sized chunks from lst."""
#     for idx in range(0, len(collection), chunk_size):
#         yield collection[idx:idx + chunk_size]

# Training images.
data_image = io.ImageCollection(load_pattern=os.path.join(train_image_nocrop, '*.png'),
                                plugin=None)
data_label = io.ImageCollection(load_pattern=os.path.join(train_label_nocrop, '*.png'),
                                plugin=None)

print(f'* Training images: {len(data_image)}; labels: {len(data_label)}')

# We need to load all data, to be able to pad the structure before operating in it.
data_image = data_image.concatenate()
data_label = data_label.concatenate()

# Then, we can pad the image at once.
data_image = np.pad(data_image, pad_width=pad_width)
data_label = np.pad(data_label, pad_width=pad_width)

# for idx, (image, label) in enumerate(zip(image_chunks(data_image), image_chunks(data_label))):
save_cropped_3d(data_image, window_shape=window_shape, step=step,
                folder=train_image_save)

save_cropped_3d(data_label, window_shape=window_shape, step=step,
                folder=train_label_save)

# Validation images.
data_image = io.ImageCollection(load_pattern=os.path.join(val_image_nocrop, '*.png'),
                                plugin=None)
data_label = io.ImageCollection(load_pattern=os.path.join(val_label_nocrop, '*.png'),
                                plugin=None)

print(f'* Validation images: {len(data_image)}; labels: {len(data_label)}')

data_image = data_image.concatenate()
data_label = data_label.concatenate()

data_image = np.pad(data_image, pad_width=pad_width)
data_label = np.pad(data_label, pad_width=pad_width)

save_cropped_3d(data_image, window_shape=window_shape, step=step,
                folder=val_image_save)

save_cropped_3d(data_label, window_shape=window_shape, step=step,
                folder=val_label_save)

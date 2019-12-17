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
folder_save = '/home/alex/data/larson_2019/data/original_288x288-3d/'

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


def save_cropped_3d(image, index, window_shape=(10, 256, 256), step=256, folder='temp'):
    """Crops image and saves the cropped chunks in disk.

    Parameters
    ----------
    image : ndarray
        Input image.
    index : int
        Reference number to saved files.
    window_shape : integer or tuple of length image.ndim, optional
        (default : (10, 256, 256))
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length image.ndim, optional (default : 256)
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
        fname = '%03d_img_crop-%03d.npz' % (index, idx)
        # io.imsave(os.path.join(folder, fname), aux)
        np.savez_compressed(os.path.join(folder, fname), aux)
    return None


data_image = io.ImageCollection(load_pattern=os.path.join(train_image_nocrop,
                                                          '*.png'),
                                plugin=None)
data_label = io.ImageCollection(load_pattern=os.path.join(train_label_nocrop,
                                                          '*.png'),
                                plugin=None)

print(f'* Training images: {len(data_image)}; labels: {len(data_label)}')

pad_width = 16  # was : 32
planes, rows, cols = (10, 256, 256)  # was : (512, 512)
window_shape = (planes, rows + 2*pad_width, cols + 2*pad_width)
step = 256  # was : 512


for idx, (image, label) in enumerate(zip(data_image, data_label)):
    image = np.pad(image, pad_width=pad_width)
    label = np.pad(label, pad_width=pad_width)

    save_cropped_3d(image, index=idx, window_shape=window_shape,
                    step=step, folder=train_image_save)

    image = np.pad(image, pad_width=pad_width)
    save_cropped_3d(label, index=idx, window_shape=window_shape,
                    step=step, folder=train_label_save)

print(f'* Validation images: {len(data_image)}; labels: {len(data_label)}')

for idx, (image, label) in enumerate(zip(data_image, data_label)):
    image = np.pad(image, pad_width=pad_width)
    label = np.pad(label, pad_width=pad_width)

    save_cropped_image(image, index=idx, window_shape=window_shape,
                       step=step, folder=val_image_save)

    save_cropped_image(label, index=idx, window_shape=window_shape,
                       step=step, folder=val_label_save)

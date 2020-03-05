from glob import glob
from itertools import chain, product
from matplotlib import mlab
from matplotlib.animation import ArtistAnimation
from scipy.ndimage.morphology import (binary_fill_holes,
                                      distance_transform_edt)
from scipy.stats import norm
from skimage import util
from skimage.color import gray2rgb
from skimage.draw import ellipse_perimeter, ellipse
from skimage.exposure import equalize_hist
from skimage.filters import threshold_multiotsu
from skimage import io, morphology
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import clear_border, mark_boundaries
from string import ascii_lowercase

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import warnings


# Setting up the figures appearance.
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

# Defining some helping variables.
OFFSET = -15
# defining some very used variables
FONT_SIZE = 22
LINE_WIDTH = 4
SAVING_EXT = 'png'
SCATTER_SIZE = 25

# The files we are going to use.
FNAME_TIRAMISU = '../coefficients/larson2019_tiramisu-67/output.train_tiramisu-67.py'
FNAME_UNET = '../coefficients/larson2019_unet/output.train_unet.py'
FNAME_UNET_3D = '../coefficients/larson2019_unet_3d/output.train_unet_3d.py'

# Determining colors for each network.
COLOR_TIRAMISU = '#3e4989'
COLOR_UNET = '#6ece58'
COLOR_TIRAMISU_3D = ''
COLOR_UNET_3D = ''




def figure_2():
    """
    Figure 2. Exemplifying the methodology using Figure 1 as the input image.
    (a) Histogram equalization and TV Chambolle's filtering (parameter:
    weight=0.3). (b) Multi Otsu's resulting regions (parameter:
    classes=4). Fibers are located within the fourth region (in yellow).
    (c) Binary image obtained considering region four in (b) as the region
    of interest, and the remaining regions as the background. (d) the
    processed region from (c), as shown in Figure 1.
    Colormaps: (a, c, d) gray, (b) viridis.
    """
    filename = 'support_figures/rec_SFRR_2600_B0p2_01000.tiff'
    image = io.imread(filename, plugin=None)
    image = util.img_as_float(image)

    # Figure 2(a).
    image_eq = equalize_hist(image)
    image_filt = denoise_tv_chambolle(image_eq, weight=0.6)

    plt.figure(figsize=(15, 10))
    plt.imshow(image_filt, cmap='gray')
    plt.savefig('figures/Fig02a.' + SAVING_EXT, bbox_inches='tight')

    # Figure 2(b).
    thresholds = threshold_multiotsu(image_filt, classes=4)
    regions = np.digitize(image_filt, bins=thresholds)

    plt.figure(figsize=(15, 10))
    plt.imshow(regions, cmap='viridis')
    plt.savefig('figures/Fig02b.' + SAVING_EXT, bbox_inches='tight')

    # Figure 2(c).
    img_fibers = remove_small_objects(regions == 3)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_fibers, cmap='gray')
    plt.savefig('figures/Fig02c.' + SAVING_EXT, bbox_inches='tight')

    # Figure 2(d).
    cut_fibers = _aux_cut_roi(img_fibers)

    plt.figure(figsize=(15, 10))
    plt.imshow(cut_fibers, cmap='gray')
    plt.savefig('figures/Fig02d.' + SAVING_EXT, bbox_inches='tight')

    # Figure 2(e).
    label_fibers, _, _ = segmentation_wusem(cut_fibers,
                                            initial_radius=0,
                                            delta_radius=2,
                                            watershed_line=True)

    permutate = np.concatenate((np.zeros(1, dtype='int'),
                                np.random.permutation(10 ** 4)),
                               axis=None)
    label_random = permutate[label_fibers]
    label_mark = mark_boundaries(
        plt.cm.nipy_spectral(label_random/label_random.max())[..., :3],
        label_img=label_random,
        color=(0, 0, 0))

    plt.figure(figsize=(15, 10))
    plt.imshow(label_mark, cmap='gist_stern')
    plt.savefig('figures/Fig02e.' + SAVING_EXT, bbox_inches='tight')

    return None


def figure_X():
    """
    Figure 1. (...).
    """

    # getting accuracy, loss for tiramisu.
    epochs_tiramisu = _aux_split_epochs(filename=FNAME_TIRAMISU)
    time_tiramisu, accuracy_tiramisu, loss_tiramisu = _aux_plot_measures(epochs_tiramisu)

    # getting accuracy, loss for unet.
    epochs_unet = _aux_split_epochs(filename=FNAME_UNET)
    time_unet, accuracy_unet, loss_unet = _aux_plot_measures(epochs_unet)


    # Fig 1 (a).
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(time_tiramisu, accuracy_tiramisu, c=COLOR_TIRAMISU, linewidth=3)
    ax[0].plot(time_unet, accuracy_unet, c=COLOR_UNET, linewidth=3)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(['Tiramisu', 'U-net'], shadow=True)

    # Fig 1 (b).
    ax[1].plot(time_tiramisu, loss_tiramisu, c=COLOR_TIRAMISU, linewidth=3)
    ax[1].plot(time_unet, loss_unet, c=COLOR_UNET, linewidth=3)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Loss')
    ax[1].legend(['Tiramisu', 'U-net'], shadow=True)

    return None





def _aux_plot_measures(epochs):
    all_time, all_accuracy, all_loss = [], [], []
    last_time = 0

    for epoch in epochs:
        accuracy, loss = _aux_epoch_indicators(epoch)
        all_accuracy.append(accuracy)
        all_loss.append(loss)

        total_time = float(epoch[-1][1].split('s')[0])
        time = np.linspace(start=last_time,
                           stop=total_time,
                           num=len(accuracy))
        all_time.append(time)
        last_time = total_time

    return np.ravel(all_time), np.ravel(all_accuracy), np.ravel(all_loss)



    """
    # now, let's get how much each epoch took.
    all_time_tiramisu, all_time_unet = [], []
    epochs = _aux_split_epochs(filename=FNAME_TIRAMISU)
    for epoch in epochs:
        all_time_tiramisu.append(float(epoch[-1][1].split('s')[0]))

    epochs = _aux_split_epochs(filename=FNAME_UNET)
    for epoch in epochs:
        all_time_unet.append(float(epoch[-1][1].split('s')[0]))

    all_time_tiramisu = np.asarray(all_time_tiramisu)
    all_time_unet = np.asarray(all_time_unet)

    print(all_time_tiramisu.mean(), all_time_tiramisu.std())
    print(all_time_unet.mean(), all_time_unet.std())

    return None
    """

def _aux_cut_roi(image, crop_roi=False):
    """
    """
    rows, cols = image.shape
    rows_c = rows // 2 - 50
    cols_c = cols // 2 - 10

    radius_a = 900
    radius_b = 935
    rr, cc = ellipse(rows_c, cols_c, radius_a, radius_b, rotation=np.pi/4)

    mask = image < 0
    mask[rr, cc] = True
    image *= mask

    if crop_roi:
        image = image[rows_c-radius_a:rows_c+radius_a,
                      cols_c-radius_b:cols_c+radius_b]
    return image


def _aux_epoch_indicators(epoch):
    """
    """
    all_accuracy, all_loss = [], []

    for row in epoch:
        aux = ' '.join(row)
        if 'accuracy:' in aux:
            #            / remove '\x08'     / separate num / remove space
            acc = row[-1].replace('\x08', '').split(':')[-1].strip()
            all_accuracy.append(float(acc))
            #             / separate num / remove space
            loss = row[-2].split(':')[-1].strip()
            all_loss.append(float(loss))

    return np.asarray(all_accuracy), np.asarray(all_loss)


def _aux_split_epochs(filename):
    """
    """
    content, idx_epochs, epochs = [], [], []

    with open(filename) as file:
        for row in csv.reader(file, delimiter='-'):
            content.append(row)

    for idx, row in enumerate(content):
        aux = ' '.join(row)
        if 'Epoch' in aux and len(aux) < 20:
            idx_epochs.append(idx)

    for idx, _ in enumerate(idx_epochs[:-1]):
        epochs.append(content[idx_epochs[idx]:idx_epochs[idx+1]])

    return epochs


def segmentation_wusem(image, str_el='disk', initial_radius=10,
                       delta_radius=5, watershed_line=False):
    """Separates regions on a binary input image using successive
    erosions as markers for the watershed algorithm. The algorithm stops
    when the erosion image does not have objects anymore.

    Parameters
    ----------
    image : (N, M) ndarray
        Binary input image.
    str_el : string, optional
        Structuring element used to erode the input image. Accepts the
        strings 'diamond', 'disk' and 'square'. Default is 'disk'.
    initial_radius : int, optional
        Initial radius of the structuring element to be used in the
        erosion. Default is 10.
    delta_radius : int, optional
        Delta radius used in the iterations:
         * Iteration #1: radius = initial_radius + delta_radius
         * Iteration #2: radius = initial_radius + 2 * delta_radius,
        and so on. Default is 5.

    Returns
    -------
    img_labels : (N, M) ndarray
        Labeled image presenting the regions segmented from the input
        image.
    num_objects : int
        Number of objects in the input image.
    last_radius : int
        Radius size of the last structuring element used on the erosion.

    References
    ----------
    .. [1] F.M. Schaller et al. "Tomographic analysis of jammed ellipsoid
    packings", in: AIP Conference Proceedings, 2013, 1542: 377-380. DOI:
    10.1063/1.4811946.

    Examples
    --------
    >>> from skimage.data import binary_blobs
    >>> image = binary_blobs(length=512, seed=0)
    >>> img_labels, num_objects, _ = segmentation_wusem(image,
                                                        str_el='disk',
                                                        initial_radius=10,
                                                        delta_radius=3)
    """
    rows, cols = image.shape
    img_labels = np.zeros((rows, cols))
    curr_radius = initial_radius
    distance = distance_transform_edt(image)

    while True:
        aux_se = {
            'diamond': morphology.diamond(curr_radius),
            'disk': morphology.disk(curr_radius),
            'square': morphology.square(curr_radius)
        }
        str_el = aux_se.get('disk', morphology.disk(curr_radius))

        erod_aux = morphology.binary_erosion(image, selem=str_el)
        if erod_aux.min() == erod_aux.max():
            last_step = curr_radius
            break

        markers = label(erod_aux)

        if watershed_line:
            curr_labels = morphology.watershed(-distance,
                                               markers,
                                               mask=image,
                                               watershed_line=True)
            img_labels += curr_labels
        else:
            curr_labels = morphology.watershed(-distance,
                                               markers,
                                               mask=image)
            img_labels += curr_labels

        # preparing for another loop.
        curr_radius += delta_radius

    # reordering labels.
    img_labels = label(img_labels)

    # removing small labels.
    img_labels, num_objects = label(remove_small_objects(img_labels),
                                    return_num=True)

    return img_labels, num_objects, last_step

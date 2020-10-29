from itertools import product
from itkwidgets import view
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from pathlib import Path
from scipy.ndimage.morphology import distance_transform_edt
from skimage import io, morphology, segmentation, util
from skimage.color import gray2rgb
from skimage.draw import ellipse
from skimage.exposure import equalize_hist
from skimage.filters import threshold_multiotsu
from skimage.measure import label
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import mark_boundaries
from sklearn.metrics import auc, roc_curve

import fullconvnets.constants as const
import itk
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import warnings

# Setting up the figures appearance.
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = 0.6*plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

# Defining some helping variables.
OFFSET = -15
SEC_TO_HOURS = 3.6E3
LINE_WIDTH = 7
SCATTER_SIZE = 25
FIGURE_SIZE = (15, 12)
BBOX_TO_ANCHOR = (0.1, 1.01)
ZOOM = 3.5

COLOR_TIRAMISU = '#ffae42'
COLOR_TIRAMISU_3D = '#ff5349'
COLOR_UNET = '#8a2be2'
COLOR_UNET_3D = '#0d98ba'
COLORS = [COLOR_TIRAMISU, COLOR_UNET, COLOR_TIRAMISU_3D, COLOR_UNET_3D]

# Setting patches for the legends.
patch_tiramisu = mpatches.Patch(color=COLOR_TIRAMISU, label='Tiramisu')
patch_tiramisu_3d = mpatches.Patch(color=COLOR_TIRAMISU_3D, label='3D Tiramisu')
patch_unet = mpatches.Patch(color=COLOR_UNET, label='U-net')
patch_unet_3d = mpatches.Patch(color=COLOR_UNET_3D, label='3D U-net')

SAVE_FIG_FORMAT = '.pdf'

# Defining network train and predict parameters.
OUTPUT_TRAIN_BASE = Path('/home/alex/pCloudDrive/data/larson_2019/coefficients/output_train')
OUTPUT_TRAIN_TIRAMISU = OUTPUT_TRAIN_BASE/'output.train_tiramisu-67.txt'
OUTPUT_TRAIN_TIRAMISU_3D = OUTPUT_TRAIN_BASE/'output.train_tiramisu_3d-67.txt'
OUTPUT_TRAIN_UNET = OUTPUT_TRAIN_BASE/'output.train_unet.txt'
OUTPUT_TRAIN_UNET_3D = OUTPUT_TRAIN_BASE/'output.train_unet_3d.txt'

OUTPUT_PREDICT_BASE = Path('/home/alex/pCloudDrive/data/larson_2019/coefficients/output_predict')
OUTPUT_PREDICT_TIRAMISU = OUTPUT_PREDICT_BASE/'output.predict_tiramisu-67.txt'
OUTPUT_PREDICT_TIRAMISU_3D = OUTPUT_PREDICT_BASE/'output.predict_tiramisu_3d-67.txt'
OUTPUT_PREDICT_UNET = OUTPUT_PREDICT_BASE/'output.predict_unet.txt'
OUTPUT_PREDICT_UNET_3D = OUTPUT_PREDICT_BASE/'output.predict_unet_3d.txt'

# Individual ROC and AUC samples.
OUTPUT_COMP_BASE = Path('comp_coefficients')
OUTPUT_COMP_TIRAMISU = OUTPUT_COMP_BASE/'tiramisu-67'
OUTPUT_COMP_TIRAMISU_3D = OUTPUT_COMP_BASE/'tiramisu_3d-67'
OUTPUT_COMP_UNET = OUTPUT_COMP_BASE/'unet'
OUTPUT_COMP_UNET_3D = OUTPUT_COMP_BASE/'unet_3d'

ROC_AUC_TIRAMISU = {'232p1_wet': OUTPUT_COMP_TIRAMISU/'rec20160324_055424_232p1_wet_1cm_cont_4097im_1500ms_17keV_13_a.h5-tiramisu_roc_auc.csv',
                    '232p3_cured': OUTPUT_COMP_TIRAMISU/'rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5-tiramisu_roc_auc.csv',
                    '232p3_wet': OUTPUT_COMP_TIRAMISU/'rec20160318_191511_232p3_2cm_cont__4097im_1500ms_ML17keV_6.h5-tiramisu_roc_auc.csv',
                    '244p1_cured': OUTPUT_COMP_TIRAMISU/'rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5-tiramisu_roc_auc.csv',
                    '244p1_wet': OUTPUT_COMP_TIRAMISU/'rec20160318_223946_244p1_1p5cm_cont__4097im_1500ms_ML17keV_7.h5-tiramisu_roc_auc.csv'}

ROC_AUC_TIRAMISU_3D = {'232p1_wet': OUTPUT_COMP_TIRAMISU_3D/'rec20160324_055424_232p1_wet_1cm_cont_4097im_1500ms_17keV_13_a.h5-tiramisu_3d_roc_auc.csv',
                       '232p3_cured': OUTPUT_COMP_TIRAMISU_3D/'rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5-tiramisu_3d_roc_auc.csv',
                       '232p3_wet': OUTPUT_COMP_TIRAMISU_3D/'rec20160318_191511_232p3_2cm_cont__4097im_1500ms_ML17keV_6.h5-tiramisu_3d_roc_auc.csv',
                       '244p1_cured': OUTPUT_COMP_TIRAMISU_3D/'rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5-tiramisu_3d_roc_auc.csv',
                       '244p1_wet': OUTPUT_COMP_TIRAMISU_3D/'rec20160318_223946_244p1_1p5cm_cont__4097im_1500ms_ML17keV_7.h5-tiramisu_3d_roc_auc.csv'}

ROC_AUC_UNET = {'232p1_wet': OUTPUT_COMP_UNET/'rec20160324_055424_232p1_wet_1cm_cont_4097im_1500ms_17keV_13_a.h5-unet_roc_auc.csv',
                '232p3_cured': OUTPUT_COMP_UNET/'rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5-unet_roc_auc.csv',
                '232p3_wet': OUTPUT_COMP_UNET/'rec20160318_191511_232p3_2cm_cont__4097im_1500ms_ML17keV_6.h5-unet_roc_auc.csv',
                '244p1_cured': OUTPUT_COMP_UNET/'rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5-unet_roc_auc.csv',
                '244p1_wet': OUTPUT_COMP_UNET/'rec20160318_223946_244p1_1p5cm_cont__4097im_1500ms_ML17keV_7.h5-unet_roc_auc.csv'}

ROC_AUC_UNET_3D = {'232p1_wet': OUTPUT_COMP_UNET_3D/'rec20160324_055424_232p1_wet_1cm_cont_4097im_1500ms_17keV_13_a.h5-unet_3d_roc_auc.csv',
                   '232p3_cured': OUTPUT_COMP_UNET_3D/'rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5-unet_3d_roc_auc.csv',
                   '232p3_wet': OUTPUT_COMP_UNET_3D/'rec20160318_191511_232p3_2cm_cont__4097im_1500ms_ML17keV_6.h5-unet_3d_roc_auc.csv',
                   '244p1_cured': OUTPUT_COMP_UNET_3D/'rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5-unet_3d_roc_auc.csv',
                   '244p1_wet': OUTPUT_COMP_UNET_3D/'rec20160318_223946_244p1_1p5cm_cont__4097im_1500ms_ML17keV_7.h5-unet_3d_roc_auc.csv'}

# Ignoring warnings.
warnings.filterwarnings('ignore')


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
    filename = 'figures/.Fig_01-original_larson.tif'
    image = io.imread(filename, as_gray=True, plugin=None)
    image = util.img_as_float(image)

    # Figure 2(a).
    image_eq = equalize_hist(image)
    image_filt = denoise_tv_chambolle(image_eq, weight=0.6)

    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(image_filt, cmap='gray')
    plt.savefig('figures/Fig_02a' + SAVE_FIG_FORMAT, bbox_inches='tight')

    # Figure 2(b).
    thresholds = threshold_multiotsu(image_filt, classes=4)
    regions = np.digitize(image_filt, bins=thresholds)

    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(regions, cmap='viridis')
    plt.savefig('figures/Fig_02b' + SAVE_FIG_FORMAT, bbox_inches='tight')

    # Figure 2(c).
    img_fibers = morphology.remove_small_objects(regions == 3)

    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(img_fibers, cmap='gray')
    plt.savefig('figures/Fig_02c' + SAVE_FIG_FORMAT, bbox_inches='tight')

    # Figure 2(d).
    cut_fibers = _cut_roi(img_fibers)

    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(cut_fibers, cmap='gray')
    plt.savefig('figures/Fig_02d' + SAVE_FIG_FORMAT, bbox_inches='tight')

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

    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(label_mark, cmap='gist_stern')

    _check_if_folder_exists(folder='./figures')
    plt.savefig('figures/Fig_02e' + SAVE_FIG_FORMAT, bbox_inches='tight')

    return None


def figure_3():
    """Figure 3. (a) Accuracy and (b) loss through time for each training epoch.
    All networks were trained during five epochs, reaching accuracy higher than
    0.9 and loss lower than 0.1 on the first training epoch, except for the
    two-dimensional U-net. However, 2D U-net is the fastest to finish training,
    and reaches the lowest loss between the candidates. We attribute the subtle
    loss increase or accuracy decrease on the start of each epoch to the data
    augmentation process.
    """
    epochs_tiramisu = _split_epochs(filename=OUTPUT_TRAIN_TIRAMISU)
    epochs_tiramisu_3d = _split_epochs(filename=OUTPUT_TRAIN_TIRAMISU_3D)
    epochs_unet = _split_epochs(filename=OUTPUT_TRAIN_UNET)
    epochs_unet_3d = _split_epochs(filename=OUTPUT_TRAIN_UNET_3D)

    accuracies_tiramisu, losses_tiramisu, time_tiramisu, _, ends_tiramisu = _return_all_measures(epochs_tiramisu)
    accuracies_tiramisu_3d, losses_tiramisu_3d, time_tiramisu_3d, _, ends_tiramisu_3d = _return_all_measures(epochs_tiramisu_3d)
    accuracies_unet, losses_unet, time_unet, _, ends_unet = _return_all_measures(epochs_unet)
    accuracies_unet_3d, losses_unet_3d, time_unet_3d, _, ends_unet_3d = _return_all_measures(epochs_unet_3d)

    # setting patches for the legends.
    patch_tiramisu = mpatches.Patch(color=COLOR_TIRAMISU, label='Tiramisu')
    patch_tiramisu_3d = mpatches.Patch(color=COLOR_TIRAMISU_3D, label='3D Tiramisu')
    patch_unet = mpatches.Patch(color=COLOR_UNET, label='U-net')
    patch_unet_3d = mpatches.Patch(color=COLOR_UNET_3D, label='3D U-net')

    # Figure 3(a).
    fig, ax = plt.subplots(nrows=2, figsize=FIGURE_SIZE)
    ax[0] = _plot_individual_measure(time_tiramisu,
                                     accuracies_tiramisu,
                                     ends=ends_tiramisu,
                                     c=COLOR_TIRAMISU,
                                     s=1,
                                     linestyle='--',
                                     ax=ax[0])
    ax[0] = _plot_individual_measure(time_unet,
                                     accuracies_unet,
                                     ends=ends_unet,
                                     c=COLOR_UNET,
                                     s=1,
                                     linestyle='--',
                                     ax=ax[0])
    ax[0].legend(handles=[patch_tiramisu,
                       patch_unet,
                       patch_tiramisu_3d,
                       patch_unet_3d],
              loc='lower left', bbox_to_anchor=BBOX_TO_ANCHOR, ncol=4,
              borderaxespad=0, frameon=False)

    ax[1] = _plot_individual_measure(time_tiramisu_3d,
                                     accuracies_tiramisu_3d,
                                     ends=ends_tiramisu_3d,
                                     c=COLOR_TIRAMISU_3D,
                                     s=1,
                                     linestyle='--',
                                     ax=ax[1])
    ax[1] = _plot_individual_measure(time_unet_3d,
                                     accuracies_unet_3d,
                                     ends=ends_unet_3d,
                                     c=COLOR_UNET_3D,
                                     s=1,
                                     linestyle='--',
                                     ax=ax[1])

    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    ax[1].set_xlabel('Time (hours)')
    plt.savefig('figures/Fig_03a' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figure 3(b).
    fig, ax = plt.subplots(nrows=2, figsize=FIGURE_SIZE)
    ax[0] = _plot_individual_measure(time_tiramisu,
                                     losses_tiramisu,
                                     ends=ends_tiramisu,
                                     c=COLOR_TIRAMISU,
                                     s=1,
                                     linestyle='--',
                                     ax=ax[0])
    ax[0] = _plot_individual_measure(time_unet,
                                     losses_unet,
                                     ends=ends_unet,
                                     c=COLOR_UNET,
                                     s=1,
                                     linestyle='--',
                                     ax=ax[0])
    ax[0].legend(handles=[patch_tiramisu,
                       patch_unet,
                       patch_tiramisu_3d,
                       patch_unet_3d],
              loc='lower left', bbox_to_anchor=BBOX_TO_ANCHOR, ncol=4,
              borderaxespad=0, frameon=False)

    ax[1] = _plot_individual_measure(time_tiramisu_3d,
                                     losses_tiramisu_3d,
                                     ends=ends_tiramisu_3d,
                                     c=COLOR_TIRAMISU_3D,
                                     s=1,
                                     linestyle='--',
                                     ax=ax[1])
    ax[1] = _plot_individual_measure(time_unet_3d,
                                     losses_unet_3d,
                                     ends=ends_unet_3d,
                                     c=COLOR_UNET_3D,
                                     s=1,
                                     linestyle='--',
                                     ax=ax[1])

    fig.text(0.02, 0.5, 'Loss', va='center', rotation='vertical')
    ax[1].set_xlabel('Time (hours)')

    _check_if_folder_exists(folder='./figures')
    plt.savefig('figures/Fig_03b' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()


def figure_4():
    """Figure 4. Accuracy vs. loss on the first epoch. Accuracy surpasses 0.9
    and loss is lower than 0.1 for all networks during the first epoch, except
    for 2D U-net (loss of 0.23). The large size of the training set and the
    similarities in the data are responsible for such numbers. Validation
    accuracy and validation loss on the first epoch are represented by diamonds.
    """
    epochs_tiramisu = _split_epochs(filename=OUTPUT_TRAIN_TIRAMISU)
    epochs_tiramisu_3d = _split_epochs(filename=OUTPUT_TRAIN_TIRAMISU_3D)
    epochs_unet = _split_epochs(filename=OUTPUT_TRAIN_UNET)
    epochs_unet_3d = _split_epochs(filename=OUTPUT_TRAIN_UNET_3D)

    accuracies_tiramisu, losses_tiramisu, _, validation_tiramisu, _ = _return_all_measures(epochs_tiramisu, concatenate=False)
    accuracies_unet, losses_unet, _, validation_unet, _ = _return_all_measures(epochs_unet, concatenate=False)
    accuracies_tiramisu_3d, losses_tiramisu_3d, _, validation_tiramisu_3d, _ = _return_all_measures(epochs_tiramisu_3d, concatenate=False)
    accuracies_unet_3d, losses_unet_3d, _, validation_unet_3d, _ = _return_all_measures(epochs_unet_3d, concatenate=False)

    # Figure 4.
    _, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax_ins = zoomed_inset_axes(ax, zoom=ZOOM-1.5, loc='upper left')
    ax = _plot_accuracy_loss(accuracies_tiramisu[0], losses_tiramisu[0],
                             validation_tiramisu[0], s=1, c=COLOR_TIRAMISU,
                             ax=ax, ax_ins=ax_ins)
    ax = _plot_accuracy_loss(accuracies_unet[0], losses_unet[0],
                             validation_unet[0], s=1, c=COLOR_UNET, ax=ax,
                             ax_ins=ax_ins)
    ax = _plot_accuracy_loss(accuracies_tiramisu_3d[0], losses_tiramisu_3d[0],
                             validation_tiramisu_3d[0][::-1], s=1,  # acc/loss inverted here!
                             c=COLOR_TIRAMISU_3D, ax=ax, ax_ins=ax_ins)
    ax = _plot_accuracy_loss(accuracies_unet_3d[0], losses_unet_3d[0],
                             validation_unet_3d[0], s=1, c=COLOR_UNET_3D, ax=ax,
                             ax_ins=ax_ins)

    ax.legend(handles=[patch_tiramisu,
                       patch_unet,
                       patch_tiramisu_3d,
                       patch_unet_3d],
              loc='lower left', bbox_to_anchor=BBOX_TO_ANCHOR, ncol=4,
              borderaxespad=0, frameon=False)

    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Loss')

    _check_if_folder_exists(folder='./figures')
    plt.savefig('figures/Fig_04' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    return None

def figure_5():
    """Figure 5. Mean and standard deviation for prediction times for each sample.
    The processing time results are similar to the training ones: 2D U-net and
    2D Tiramisu are the fastest architectures to process a sample, predicting in
    a whole sample in one hour, in average. 3D Tiramisu, being the slowest, takes
    in average more than a day to process one sample.
    """
    time_tiramisu = _split_predictions(filename=OUTPUT_PREDICT_TIRAMISU)
    time_unet = _split_predictions(filename=OUTPUT_PREDICT_UNET)
    time_tiramisu_3d = _split_predictions(filename=OUTPUT_PREDICT_TIRAMISU_3D)
    time_unet_3d = _split_predictions(filename=OUTPUT_PREDICT_UNET_3D)

    sum_tiramisu = _sum_duration_predictions(time_tiramisu)
    sum_unet = _sum_duration_predictions(time_unet)
    sum_tiramisu_3d = _sum_duration_predictions(time_tiramisu_3d)
    sum_unet_3d = _sum_duration_predictions(time_unet_3d)

    width = 0.4  # the width of the bars

    # Figure 5.
    sum_networks = [sum_tiramisu,
                    sum_unet,
                    sum_tiramisu_3d,
                    sum_unet_3d]
    bars_range = np.arange(len(sum_networks))

    x_ticks = bars_range
    x_labels = ['Tiramisu', 'U-net', '3D Tiramisu', '3D U-net']

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    # ax.set_yscale('log')

    for idx in bars_range:
        aux_mean = sum_networks[idx].mean() / 3600
        aux_std = sum_networks[idx].std() / 3600

        # adjusting significant figures
        aux_std = np.round(aux_std+np.modf(aux_mean)[0], decimals=2)
        aux_mean = int(aux_mean)

        ax.bar(idx, aux_mean, width, yerr=aux_std, label=x_labels,
               color=COLORS[idx])
        if idx in (bars_range[-1], bars_range[-2]):
            ax.annotate(f'{aux_mean}±{aux_std}',
                        xy=(idx - width / 1.5, aux_mean),
                        ha='center', va='bottom')
        else:
            ax.annotate(f'{aux_mean}±{aux_std}',
                        xy=(idx + width / 1.5, aux_mean),
                        ha='center', va='bottom')
    ax.set_ylabel('Time (hours)')

    # Checking if the folder 'figures' exists.
    _check_if_folder_exists(folder='./figures')
    plt.savefig('figures/Fig_05' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    return None


def figure_6():
    """ Figure 6. Receiver operating characteristic (ROC) and area under curve
    (AUC) from the comparison between the prediction for each network and the
    segmentation made available for five samples by Larson et al. (2019). ROC
    curves were calculated to all slices in a dataset; their mean areas and
    standard deviation intervals are presented. AUC is larger than 98% in all
    comparisons, showing that our predictions are accurate when compared with
    Larson et al. semi-supervised method. The 2D versions of U-net and Tiramisu
    perform better when compared to their 3D alternatives.
    """
    # Figure 6 (a).
    SAMPLE_232p1_wet = [ROC_AUC_TIRAMISU['232p1_wet'],
                        ROC_AUC_UNET['232p1_wet'],
                        ROC_AUC_TIRAMISU_3D['232p1_wet'],
                        ROC_AUC_UNET_3D['232p1_wet']]
    _, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax_ins = zoomed_inset_axes(ax, zoom=ZOOM, loc='lower right')
    area_under_curve = []
    for idx, filename in enumerate(SAMPLE_232p1_wet):
        fp_rate, tp_rate, auc_mean = _read_csv_roc_auc(filename)
        area_under_curve.append(auc_mean)
        _plot_roc_and_auc(fp_rate, tp_rate, c=COLORS[idx],
                          linestyle='-', ax=ax, ax_ins=ax_ins)

    ax = _add_auc_legend(area_under_curve, ax=ax)
    plt.savefig('figures/Fig_06a' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figure 6 (b).
    SAMPLE_232p3_cured = [ROC_AUC_TIRAMISU['232p3_cured'],
                          ROC_AUC_UNET['232p3_cured'],
                          ROC_AUC_TIRAMISU_3D['232p3_cured'],
                          ROC_AUC_UNET_3D['232p3_cured']]
    _, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax_ins = zoomed_inset_axes(ax, zoom=ZOOM, loc='lower right')
    area_under_curve = []
    for idx, filename in enumerate(SAMPLE_232p3_cured):
        fp_rate, tp_rate, auc_mean = _read_csv_roc_auc(filename)
        area_under_curve.append(auc_mean)
        _plot_roc_and_auc(fp_rate, tp_rate, c=COLORS[idx],
                          linestyle='-', ax=ax, ax_ins=ax_ins)

    ax = _add_auc_legend(area_under_curve, ax=ax)
    plt.savefig('figures/Fig_06b' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figure 6 (c).
    SAMPLE_232p3_wet = [ROC_AUC_TIRAMISU['232p3_wet'],
                        ROC_AUC_UNET['232p3_wet'],
                        ROC_AUC_TIRAMISU_3D['232p3_wet'],
                        ROC_AUC_UNET_3D['232p3_wet']]
    _, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax_ins = zoomed_inset_axes(ax, zoom=ZOOM, loc='lower right')
    area_under_curve = []
    for idx, filename in enumerate(SAMPLE_232p3_wet):
        fp_rate, tp_rate, auc_mean = _read_csv_roc_auc(filename)
        area_under_curve.append(auc_mean)
        _plot_roc_and_auc(fp_rate, tp_rate, c=COLORS[idx],
                          linestyle='-', ax=ax, ax_ins=ax_ins)

    ax = _add_auc_legend(area_under_curve, ax=ax)
    plt.savefig('figures/Fig_06c' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figure 6 (d).
    SAMPLE_244p1_cured = [ROC_AUC_TIRAMISU['244p1_cured'],
                          ROC_AUC_UNET['244p1_cured'],
                          ROC_AUC_TIRAMISU_3D['244p1_cured'],
                          ROC_AUC_UNET_3D['244p1_cured']]
    _, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax_ins = zoomed_inset_axes(ax, zoom=ZOOM, loc='lower right')
    area_under_curve = []
    for idx, filename in enumerate(SAMPLE_244p1_cured):
        fp_rate, tp_rate, auc_mean = _read_csv_roc_auc(filename)
        area_under_curve.append(auc_mean)
        _plot_roc_and_auc(fp_rate, tp_rate, c=COLORS[idx],
                          linestyle='-', ax=ax, ax_ins=ax_ins)

    ax = _add_auc_legend(area_under_curve, ax=ax)
    plt.savefig('figures/Fig_06d' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figure 6 (e).
    SAMPLE_244p1_wet = [ROC_AUC_TIRAMISU['244p1_wet'],
                        ROC_AUC_UNET['244p1_wet'],
                        ROC_AUC_TIRAMISU_3D['244p1_wet'],
                        ROC_AUC_UNET_3D['244p1_wet']]
    _, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax_ins = zoomed_inset_axes(ax, zoom=ZOOM, loc='lower right')
    area_under_curve = []
    for idx, filename in enumerate(SAMPLE_244p1_wet):
        fp_rate, tp_rate, auc_mean = _read_csv_roc_auc(filename)
        area_under_curve.append(auc_mean)
        _plot_roc_and_auc(fp_rate, tp_rate, c=COLORS[idx],
                          linestyle='-', ax=ax, ax_ins=ax_ins)

    ax = _add_auc_legend(area_under_curve, ax=ax)
    _check_if_folder_exists(folder='./figures')
    plt.savefig('figures/Fig_06e' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    return None


def figure_8():
    """


    Notes
    -----
    Colors from Bang Wong's color-blind friendly colormap. Available at:
    https://www.nature.com/articles/nmeth.1618

    Wong's map acquired from David Nichols page. Available at:
    https://davidmathlogic.com/colorblind/.
    """
    # choosing test sample and network.
    sample = const.SAMPLE_232p3_wet
    network_folder = const.FOLDER_PRED_UNET

    # we will return a 10 x 10 matthews matrix; each for a crop
    matthews_coefs = np.ones((10, 10))
    worst_indexes = np.zeros((10, 10))

    # a variable to obtain inlay data.
    inlay_data = []

    # reading input data.
    is_registered = sample['registered_path'] is not None
    data_pred, data_gs = _pred_and_goldstd(sample,
                                           folder_prediction=network_folder,
                                           is_registered=is_registered,
                                           is_binary=True)
    data_pred = data_pred[slice(*sample['segmentation_interval'])]

    # comp_color starts as gray (background).
    comp_color = np.ones(
        (*data_pred[0].shape, 3)
    ) * (np.asarray((238, 238, 238)) / 255)

    for idx, (img_pred, img_gs) in enumerate(zip(data_pred, data_gs)):
        # crop images in 100 (256, 256) pieces.
        crop_pred = util.view_as_blocks(img_pred,
                                        block_shape=(256, 256))
        crop_gs = util.view_as_blocks(img_gs,
                                      block_shape=(256, 256))
        for i, _ in enumerate(crop_pred):
            for j, _ in enumerate(crop_pred[i]):

                # calculate the Matthews coefficient for each crop.
                aux_conf = _confusion_matrix(crop_gs[i, j],
                                             crop_pred[i, j])
                aux_matthews = _measure_matthews(aux_conf)

                # if smaller than previously, save results.
                # restricting aux_matthews > 0.1 due to errors in all-TN regions
                if (0.1 < aux_matthews < matthews_coefs[i, j]):
                    matthews_coefs[i, j] = aux_matthews
                    worst_indexes[i, j] = idx

                    aux_comp = _comparison_color(crop_gs[i, j], crop_pred[i, j])
                    comp_color[i*256:(i+1)*256, j*256:(j+1)*256] = aux_comp

    # grab inlay data from crops we want to highlight.
    for i, j in [(2, 2), (8, 7)]:
        inlay_data.append(comp_color[i*256:(i+1)*256, j*256:(j+1)*256])

    # Figure 8(a).
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(comp_color)

    for idx in np.arange(start=0, stop=2560, step=256):  # according to image
        plt.axvline(idx, color='white')
        plt.axhline(idx, color='white')

    matthews_coefs = np.round(matthews_coefs * 100, decimals=2)

    for i, j in product(range(10), repeat=2):
        facecolor, textcolor = _label_color(matthews_coefs[j, i])
        plt.text(x=i*256 + 30, y=j*256 + 50,
                 s=str(matthews_coefs[j, i]),
                 fontsize=8,
                 color=textcolor,
                 bbox=dict(facecolor=facecolor, alpha=0.9))

    _check_if_folder_exists(folder='./figures')
    plt.savefig('figures/Fig_08a' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figures 8(b, c).
    indexes = {0: 'b', 1: 'c'}
    for idx in indexes.keys():
        plt.figure(figsize=FIGURE_SIZE)
        plt.imshow(inlay_data[idx])

        _check_if_folder_exists(folder='./figures')
        plt.savefig(f'figures/Fig_08{indexes[idx]}' + SAVE_FIG_FORMAT,
                    bbox_inches='tight')
        plt.close()

    return None


def figure_9():
    """

    Notes
    -----
    You should run this code inside a Jupyter Notebook.
    """

    # reading all images from the sample.
    pattern = 'unet/rec20160318_191511_232p3_2cm_cont__4097im_1500ms_ML17keV_6.h5/predict/*.png'
    images = io.ImageCollection(pattern)
    images = images.concatenate()[:, ::2, ::2]

    # slicing the outer part to show inside fibers.
    images = images[:, :1000, :]
    images = itk.GetImageFromArray(images)

    view(images, label_image_blend=0, gradient_opacity=0, shadow=False,
         interpolation=False)

    return None


def _label_color(coef):
    """
    """
    colors = {100: '#ffffff', 90: '#fff5f0', 80: '#fee0d2', 70: '#fcbba1',
              60: '#fc9272', 50: '#fb6a4a', 40: '#ef3b2c', 30: '#cb181d',
              20: '#a50f15', 10: '#67000d', 0: '#000000'}

    percentage = np.asarray(list(colors.keys())) 
    min_perc = min(percentage[percentage - coef >= 0])

    if min_perc > 50:
        textcolor = 'black'
    else:
        textcolor = 'white'

    return colors[min_perc], textcolor


def _comparison_color(data_gs, data_pred):
    """
    """
    # comp_color starts as gray (background).
    comparison = np.ones(
        (*data_gs.shape, 3)
    ) * (np.asarray((238, 238, 238)) / 255)

    # calculating true positives, false positives, and false negatives.
    true_pos = (data_gs & data_pred)
    false_pos = (~data_gs & data_pred)
    false_neg = (data_gs & ~data_pred)

    # defining colors for the plot.
    color_fp = np.asarray((213, 94, 0)) / 255  # vermillion
    color_tp = np.asarray((0, 158, 115)) / 255  # bluish green
    color_fn = np.asarray((230, 159, 0)) / 255  # orange

    # determining where comparison will take TP, FP, and FN values.
    np.copyto(comparison, color_fp, where=false_pos[..., np.newaxis])
    np.copyto(comparison, color_tp, where=true_pos[..., np.newaxis])
    np.copyto(comparison, color_fn, where=false_neg[..., np.newaxis])

    return comparison


def _confusion_matrix(data_gs, data_pred):
    """Compares reference and test data to generate a confusion matrix.

    Parameters
    ----------
    data_gs : ndarray
        Reference binary data (ground truth).
    data_pred : ndarray
        Test binary data.

    Returns
    -------
    conf_matrix : array
        Matrix containing the number of true positives, false positives,
    false negatives, and true negatives.

    Notes
    -----
    The values true positive, false positive, false negative, and false
    positive are events obtained in the comparison between data_gs and
    data_pred:

                   data_gs:             True                False
    data_pred:
                            True       True positive   |   False positive
                                       ----------------------------------
                            False      False negative  |    True negative

    References
    ----------
    .. [1] Fawcett T. (2006) "An Introduction to ROC Analysis." Pattern
           Recognition Letters, 27 (8): 861-874, :DOI:`10.1016/j.patrec.2005.10.010`
    .. [2] Google Developers. "Machine Learning Crash Course with TensorFlow
           APIs: Classification: True vs. False and Positive vs. Negative."
           Available at:
           https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative
    .. [3] Wikipedia. "Confusion matrix." Available at:
           https://en.wikipedia.org/wiki/Confusion_matrix
    """
    true_pos = (data_gs & data_pred).sum() / data_gs.size
    false_pos = (~data_gs & data_pred).sum() / data_gs.size
    false_neg = (data_gs & ~data_pred).sum() / data_gs.size
    true_neg = (~data_gs & ~data_pred).sum() / data_gs.size

    return np.array([[true_pos, false_pos], [false_neg, true_neg]])


def _measure_matthews(conf_matrix):
    """Calculate the Matthews correlation coefficient for a confusion matrix.

    Parameters
    ----------
    conf_matrix : array
        Matrix containing the number of true positives, false positives,
    false_negatives, and true negatives.

    Returns
    -------
    coef_matthews : float
        Matthews correlation index for the input confusion matrix.

    Notes
    -----
    Matthews correlation coefficient tends to be more informative than other
    confusion matrix measures, such as matthews coefficient and accuracy, when
    evaluating binary classification problems: it considers the balance ratios
    of true positives, true negatives, false positives, and false negatives.

    References
    ----------
    .. [1] Matthews B. W. (1975) "Comparison of the predicted and observed
           secondary structure of T4 phage lysozyme." Biochimica et Biophysica
           Acta (BBA) - Protein Structure, 405 (2): 442-451,
           :DOI:`10.1016/0005-2795(75)90109-9`
    .. [2] Chicco D. (2017) "Ten quick tips for machine learning in computational
           biology." BioData Mining, 10 (35): 35, :DOI:`10.1186/s13040-017-0155-3`
    .. [3] Wikipedia. "Matthews correlation coefficient." Available at:
           https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    Examples
    --------
    >>> from skimage.measures import confusion_matrix
    >>> conf_matrix = confusion_matrix(data_true, data_test)
    >>> coef_matthews = measure_matthews(conf_matrix)
    """
    tr_pos, fl_pos, fl_neg, tr_neg = conf_matrix.ravel()
    coef_matthews = (tr_pos * tr_neg - fl_pos * fl_neg) / \
        np.sqrt((tr_pos + fl_pos) * (tr_pos + fl_neg) *
                (tr_neg + fl_pos) * (tr_neg + fl_neg))
    return coef_matthews


def _pred_and_goldstd(sample, folder_prediction, is_registered=False,
                      is_binary=True):
    """
    """
    aux_folder = sample['folder']
    if is_registered:
        aux_folder += '_REG'
        aux_subfolder = const.SUBFOLDER_GOLDSTD_REG
    else:
        aux_subfolder = const.SUBFOLDER_GOLDSTD

    folder_pred = os.path.join(folder_prediction,
                               aux_folder,
                               const.SUBFOLDER_PRED,
                               f'*{const.EXT_PRED}')
    folder_goldstd = os.path.join(const.FOLDER_GOLDSTD,
                                  sample['folder'],
                                  aux_subfolder,
                                  f'*{const.EXT_GOLDSTD}')

    data_prediction = io.ImageCollection(load_pattern=folder_pred,
                                         load_func=imread_prediction,
                                         is_binary=is_binary)
    data_goldstd = io.ImageCollection(load_pattern=folder_goldstd,
                                      load_func=imread_goldstd)
    return data_prediction, data_goldstd


def imread_prediction(image, is_binary=True):
    """Auxiliary function intended to be used with skimage.io.ImageCollection.
    Returns a binary prediction image — True when image > 0.5, False
    when image <= 0.5.
    """
    if is_binary:
        return util.img_as_float(io.imread(image)) > 0.5
    else:
        return util.img_as_float(io.imread(image))


def imread_goldstd(image):
    """Auxiliary function intended to be used with skimage.io.ImageCollection.
    Helps to read and process Larson et al's gold standard images.
    """
    data = io.imread(image)

    if np.unique(data).size > 2:
        data = data == 217

    return util.img_as_bool(data)


def _add_auc_legend(area_under_curve, ax=None):
    """
    """
    if ax is None:
        _, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Setting patches for the legends.
    _ = np.round(np.asarray(area_under_curve[0]) * 100, decimals=4)
    aux_label = f'Tiramisu (AUC$_\mu$: {_}%)'
    leg_auc_tiramisu = mpatches.Patch(color=COLOR_TIRAMISU,
                                      label=aux_label)

    _ = np.round(np.asarray(area_under_curve[1]) * 100, decimals=4)
    aux_label = f'U-net (AUC$_\mu$: {_}%)'
    leg_auc_unet = mpatches.Patch(color=COLOR_UNET,
                                  label=aux_label)

    _ = np.round(np.asarray(area_under_curve[2]) * 100, decimals=4)
    aux_label = f'3D Tiramisu (AUC$_\mu$: {_}%)'
    leg_auc_tiramisu_3d = mpatches.Patch(color=COLOR_TIRAMISU_3D,
                                         label=aux_label)

    _ = np.round(np.asarray(area_under_curve[3]) * 100, decimals=4)
    aux_label = f'3D U-net (AUC$_\mu$: {_}%)'
    leg_auc_unet_3d = mpatches.Patch(color=COLOR_UNET_3D,
                                     label=aux_label)

    ax.legend(handles=[leg_auc_tiramisu,
                       leg_auc_unet,
                       leg_auc_tiramisu_3d,
                       leg_auc_unet_3d],
              loc='lower left', bbox_to_anchor=BBOX_TO_ANCHOR, ncol=2,
              borderaxespad=0, frameon=False)
    return ax


def _check_if_folder_exists(folder: str) -> None:
    """Auxiliary function. Check if folder exists and create it if necessary."""
    if not os.path.isdir(folder):
        os.mkdir(folder)
    return None


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
            curr_labels = segmentation.watershed(-distance,
                                                 markers,
                                                 mask=image,
                                                 watershed_line=True)
            img_labels += curr_labels
        else:
            curr_labels = segmentation.watershed(-distance,
                                                 markers,
                                                 mask=image)
            img_labels += curr_labels

        # preparing for another loop.
        curr_radius += delta_radius

    # reordering labels.
    img_labels = label(img_labels)

    # removing small labels.
    img_labels, num_objects = label(morphology.remove_small_objects(img_labels),
                                    return_num=True)

    return img_labels, num_objects, last_step


def _cut_roi(image, crop_roi=False):
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


def _plot_accuracy_loss(x, y, validation, c, s=1, ax=None, ax_ins=None):
    """
    """
    if ax is None:
        _, ax = plt.subplots(figsize=FIGURE_SIZE)
    if ax_ins is None:
        ax_ins = zoomed_inset_axes(ax, zoom=2.5, loc=2)

    ax.scatter(x, y, s=s, c=c, rasterized=True)
    ax.scatter(validation[0], validation[1], c=c, s=s*50, marker='D',
               edgecolors='w', rasterized=True)
    ax_ins.scatter(x, y, s=s*2, c=c, rasterized=True)
    ax_ins.scatter(validation[0], validation[1], c=c, s=s*100, marker='D',
                   edgecolors='w', rasterized=True)

    # defining plot limits.
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.invert_yaxis()

    # defining inset limits.
    ax_ins.set_xlim([0.8, 1])
    ax_ins.set_ylim([0, 0.2])
    ax_ins.set_xticks([1])
    ax_ins.set_yticks([0.1])
    ax_ins.yaxis.tick_right()
    ax_ins.invert_yaxis()

    # adding y=1-x.
    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(limits, 1-np.asarray(limits), c='k', linestyle='--',
            rasterized=True)

    return ax


def _plot_individual_measure(x, y, ends, c, s=1, linestyle='--', ax=None):
    """
    """
    if ax is None:
        _, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.scatter(x / SEC_TO_HOURS, y, s=s, c=c, rasterized=True)
    for idx, point in enumerate(ends):
        ax.axvline(x=point[0] / SEC_TO_HOURS, c=c, linestyle=linestyle,
                   rasterized=True)
        ax.text(point[0] / SEC_TO_HOURS, 0.6, f"Epoch {idx+2}", fontsize=12,
                c=c, clip_box=ax.clipbox, va='center', rotation='vertical',
                clip_on=True, rasterized=True)
    return ax


def _plot_roc_and_auc(fpr, tpr, c, linestyle='-', ax=None, ax_ins=None):
    """
    Parameters
    ----------
    fpr : array-like
        Means and standard deviations of the false positive rate.
    tpr : array-like
        Means and standard deviations of the true positive rate.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    if ax_ins is None:
        ax_ins = zoomed_inset_axes(ax, zoom=6, loc='lower right')

    ax.plot([0, 1], [0, 1], c='k', linestyle='--', rasterized=True)
    ax.plot(fpr[0], tpr[0], c=c, linestyle=linestyle, rasterized=True)
    ax.fill_between(fpr[0], tpr[0]-tpr[1], tpr[0]+tpr[1], color=c, alpha=0.07,
                    rasterized=True)

    ax_ins.plot(fpr[0], tpr[0], c=c, linestyle=linestyle, rasterized=True)
    ax_ins.fill_between(fpr[0], tpr[0]-tpr[1], tpr[0]+tpr[1], color=c,
                        alpha=0.07, rasterized=True)

    # defining plot limits.
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # defining inset limits.
    ax_ins.set_xlim([0, 0.1])
    ax_ins.set_ylim([0.9, 1.05])
    ax_ins.set_xticks([0.05])
    ax_ins.set_yticks([1])
    ax_ins.xaxis.tick_top()

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    return ax


def _split_epochs(filename):
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


def _split_predictions(filename):
    """
    """
    content, idx_preds, aux_preds, info_steps = [], [], [], []

    with open(filename) as file:
        for row in csv.reader(file):
            content.append(row)

    for idx, row in enumerate(content):
        aux = ' '.join(row)
        if '# Processing' in aux:
            idx_preds.append(idx)

    for idx, _ in enumerate(idx_preds[:-1]):
        aux_preds.append(content[idx_preds[idx]:idx_preds[idx+1]])

    for pred in aux_preds:
        aux = []
        for row in pred:
            if '/step' in ' '.join(row):
                aux.append(row)
        info_steps.append(aux)

    time_predictions = _extract_time(info_steps)

    return time_predictions


def _sum_duration_predictions(time_predictions):
    """
    """
    sum_predictions = []

    for time in time_predictions:
        sum_predictions.append(np.asarray(time).sum())

    return np.asarray(sum_predictions)


def _extract_time(info_steps):
    """
    """
    time_predictions = []

    for sample in info_steps:
        aux = []
        for row in sample:
            # line formats are:
            # ['100/100 [==============================] ', ' 2s 16ms/step']
            # then, aux_time here will be ' 2s 16ms/step'
            aux_time = ' '.join(row).split('-')[-1]
            # splitting seconds and milliseconds
            aux_time = aux_time.split('ms')[0].split('s')
            # now, converting ms to seconds and adding the result
            aux_time = float(aux_time[0]) + float(aux_time[1]) * 1E-3
            aux.append(aux_time)

        time_predictions.append(np.asarray(aux))

    return time_predictions

def _epoch_measures(epoch):
    """
    """
    accuracies, losses = [], []

    for row in epoch:
        for elem in row:
            # need to remove '\x08' from EOL, '0.XXXXFound' from validation
            if elem.split()[0] == 'accuracy:':
                aux = elem.split()[1].replace('\x08', '').split('F')[0]
                accuracies.append(float(aux))
            elif elem.split()[0] == 'loss:':
                aux = elem.split()[1].replace('\x08', '').split('F')[0]
                losses.append(float(aux))

    return np.asarray(accuracies), np.asarray(losses)


def _read_csv_roc_auc(filename):
    """Reads csv ROC and AUC saved in a file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    fp_rate : ndarray
        Arrays containing mean and standard deviation of false positive rate for
        the processed samples.
    tp_rate : array
        Arrays containing mean and standard deviation of true positive rate for
        the processed samples.
    area_under_curve : float
        Area under curve obtained from fp_rate and tp_rate means.
    """
    coefs, fp, tp = [[] for _ in range(3)]
    csv_file = csv.reader(open(filename, 'r'))
    for row in csv_file:
        coefs.append(row)

    fp_rate, tp_rate, area_under_curve = coefs
    for (aux_fp, aux_tp) in zip(fp_rate, tp_rate):
        fp.append(aux_fp.replace('[', ' ').replace(']', ' ').split())
        tp.append(aux_tp.replace('[', ' ').replace(']', ' ').split())

    fp_rate = np.asarray(fp[1:], dtype='float')
    tp_rate = np.asarray(tp[1:], dtype='float')
    area_under_curve = np.asarray(area_under_curve[1:], dtype='float')

    return fp_rate, tp_rate, area_under_curve[0]


def _return_all_measures(epochs, concatenate=True):
    """
    """
    accuracies, losses, time, end_of_epochs, validation_measures = [[] for _ in range(5)]
    last_time = 0
    for epoch in epochs:
        # appending accuracies and losses for each epoch.
        accuracy, loss = _epoch_measures(epoch)
        accuracies.append(accuracy)
        losses.append(loss)

        # acquiring total time and creating time vector from it.
        total_time = float(epoch[-1][1].split('s')[0])
        time.append(np.linspace(start=last_time,
                                stop=total_time+last_time,
                                num=len(accuracy)))
        last_time += total_time

        # acquiring validation accuracy and loss.
        validation_measures.append([float(epoch[-1][-1].split(':')[1]),  # val_accuracy
                                    float(epoch[-1][-2].split(':')[1])])  # val_loss

        # storing last accuracy and loss point.
        end_of_epochs.append([last_time,
                              accuracy[-1],
                              loss[-1]])

    if concatenate:
        losses = np.concatenate(losses)
        accuracies = np.concatenate(accuracies)
        time = np.concatenate(time)

    return accuracies, losses, time, validation_measures, end_of_epochs


def generate_all_figures():

    import generate_figures

    print('Generating all figures...')

    for function in dir(generate_figures):
        if function.startswith('figure'):
            item = getattr(generate_figures, function)
            if callable(item):
                item()

    return None


if __name__ == '__main__':
    generate_all_figures()

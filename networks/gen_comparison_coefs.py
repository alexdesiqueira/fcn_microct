from skimage import io

import misc
import os
import utils


# folder structure generated from our results (PRED) and Larson's gold
# standard (GOLD_STD).
FOLDER_BASE = '/home/alex/data/larson_2019/'
FOLDER_PRED = 'data/unet/'
SUBFOLDER_PRED = 'predict'

FOLDER_GS = 'Seg/Bunch2/'
SUBFOLDER_GS = '19_Gray_image'


# defining prediction and gold standard file extensions.
FEXT_PRED = '*.png'
FEXT_GS = '*.tif'

# each sample defined by Larson has a specific segmentation interval.
SEGMENTATION_INTERVALS = {
    '244p1': [150, 1150],
    '232p3': [159, 1159],
    '232p1': [160, 1160],
    '235p1': None,
    '235p4': None,
    '245p1': None,
    '245p2': [1000, 2000],
    '234p3': [1000, 2000],
    '234p5': [1000, 2000],
    '234p2': None,
    '234p1': [1000, 2000],
    '234p4': None,
    '245p3': [1000, 2000]
}


def _imread_prediction(image):
    return io.imread(image) > 127


def _imread_goldstd(image):
    return utils.process_gt_images(io.imread(image))


# reading folder samples to being processed.
pred_folders, gold_folders = misc.folders_to_process(folder_base=FOLDER_BASE,
                                                     folder_pred=FOLDER_PRED,
                                                     subfolder_pred=SUBFOLDER_PRED,
                                                     folder_gold_std=FOLDER_GS,
                                                     subfolder_gold_std=SUBFOLDER_GS)

print(pred_folders, gold_folders)  # DEBUGGING

# using io.ImageCollection to read prediction and gold standard images.
for pred_folder, gold_folder in zip(pred_folders, gold_folders):
    pred_data = io.ImageCollection(load_pattern=os.path.join(pred_folder,
                                                             FEXT_PRED),
                                   load_func=_imread_prediction)
    gold_data = io.ImageCollection(load_pattern=os.path.join(gold_folder,
                                                             FEXT_GS),
                                   load_func=_imread_goldstd)

    print(len(pred_data), len(gold_data))  # DEBUGGING

    # getting the name of the sample from the path.
    name_sample = pred_folder.split('_')[3]
    slicing_interval = SEGMENTATION_INTERVALS[name_sample]
    pred_data = pred_data[slice(*slicing_interval)]

    print(name_sample, slicing_interval, len(pred_data))  # DEBUGGING

    # measuring coefficients for all data.
    filename = f'{name_sample}_coefs.csv'
    _, _ = misc.measure_all_coefficients(pred_data,
                                         gold_data,
                                         save_coef=True,
                                         filename=filename)

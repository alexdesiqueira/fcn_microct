from skimage import io

import constants as const
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
    'rec20160318_223946_244p1_1p5cm_cont__4097im_1500ms_ML17keV_7.h5': [150, 1150],
    'rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5': [150, 1150],
    'rec20160318_191511_232p3_2cm_cont__4097im_1500ms_ML17keV_6.h5': [0, 1000],
    'rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5': [0, 1000],
    'rec20160324_055424_232p1_wet_1cm_cont_4097im_1500ms_17keV_13_a.h5': [160, 1160],
    'rec20160324_123639_235p1_wet_0p7cm_cont_4097im_1500ms_17keV_14.h5': None,
    'rec20160326_175540_235p4_wet_1p15cm_cont_4097im_1500ex_17keV_20.h5': None,
    'rec20160327_003824_235p4_cured_1p15cm_cont_4097im_1500ex_17keV_22.h5': None,
    'rec20160327_160624_245p1_wet_1cm_cont_4097im_1500ex_17keV_23.h5': None
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

# using io.ImageCollection to read prediction and gold standard images.
for pred_folder, gold_folder in zip(pred_folders, gold_folders):
    pred_data = io.ImageCollection(load_pattern=os.path.join(pred_folder,
                                                             FEXT_PRED),
                                   load_func=_imread_prediction)
    gold_data = io.ImageCollection(load_pattern=os.path.join(gold_folder,
                                                             FEXT_GS),
                                   load_func=_imread_goldstd)

    # using the name of the folder to get the slicing interval.
    folder = pred_folder.split('/')[-2]
    slicing_interval = SEGMENTATION_INTERVALS[folder]
    pred_data = pred_data[slice(*slicing_interval)]

    # coefficients will receive the folder name as a filename.
    filename = f'coefs/{pred_folder.split("/")[-2]}_coefs.csv'

    # measuring coefficients for all data.
    _, _ = misc.measure_all_coefficients(pred_data,
                                         gold_data,
                                         save_coef=True,
                                         filename=filename)

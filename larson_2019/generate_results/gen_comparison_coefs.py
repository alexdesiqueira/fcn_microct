from skimage import io

import misc
import os


# folder structure generated from our results (PRED) and Larson's gold
# standard (GOLD_STD).
FOLDER_BASE = '~/data/larson_2019/'
FOLDER_PRED = 'data/unet/'
SUBFOLDER_PRED = 'predict'
FOLDER_GS = 'Seg/Bunch2/'
SUBFOLDER_GS = '19_Gray_image'

# reading folder samples to being processed.
pred_folders, gold_folders = misc.folders_to_process(folder_base=FOLDER_BASE,
                                                     folder_pred=FOLDER_PRED,
                                                     subfolder_pred=SUBFOLDER_PRED,
                                                     folder_gold_std=FOLDER_GS,
                                                     subfolder_gold_std=SUBFOLDER_GS)

# using io.ImageCollection to read prediction and gold standard images.
for pred_folder, gold_folder in zip(pred_folders, gold_folders):
    pred_data = io.ImageCollection(load_pattern=os.path.join(pred_folder, '*.tif'))
    gold_data = io.ImageCollection(load_pattern=os.path.join(gold_folder, '*.tif'))

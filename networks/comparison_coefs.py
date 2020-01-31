import constants as const
import  misc
import utils
from skimage import io

import os


NETWORK = 'unet'  # available: 'unet', 'tiramisu'
NETWORK_FOLDER = {'unet': const.FOLDER_PRED_UNET,
                  'tiramisu': const.FOLDER_PRED_TIRAMISU,
                  '3d_unet': const.FOLDER_PRED_3DUNET}

# reading folder samples to being processed.
pred_folders, gold_folders = misc.folders_to_process(folder_base=const.FOLDER_BASE,
                                                     folder_pred=const.FOLDER_PRED_UNET,
                                                     subfolder_pred=const.SUBFOLDER_PRED,
                                                     folder_gold_std=const.FOLDER_GOLDSTD,
                                                     subfolder_gold_std=const.SUBFOLDER_GOLDSTD)


for sample in const.SAMPLES_BUNCH2:
    if sample['has_goldstd']:
        pass







def _aux_read_data(sample, folder_prediction=NETWORK_FOLDER[NETWORK],
                   is_registered=False):
    """
    """
    folder_pred = os.path.join(folder_prediction,
                               sample['folder'],
                               const.SUBFOLDER_PRED,
                               '*' + const.EXT_PRED)
    if is_registered:
        folder_goldstd = os.path.join(const.FOLDER_GOLDSTD,
                                      const.SUBFOLDER_GOLDSTD,
                                      '*' + const.EXT_GOLDSTD)
    else:
        folder_goldstd = os.path.join(const.FOLDER_GOLDSTD,
                                      const.SUBFOLDER_GOLDSTD_REG,
                                      '*' + const.EXT_GOLDSTD)

    data_prediction = io.ImageCollection(load_pattern=folder_pred,
                                         load_func=utils._imread_prediction)
    data_goldstd = io.ImageCollection(load_pattern=folder_goldstd,
                                      load_func=utils._imread_goldstd)
    return data_prediction, data_goldstd


# using io.ImageCollection to read prediction and gold standard images.
for pred_folder, gold_folder in zip(pred_folders, gold_folders):
    pred_data = io.ImageCollection(load_pattern=os.path.join(pred_folder,
                                                             const.EXT_PRED),
                                   load_func=utils._imread_prediction)
    gold_data = io.ImageCollection(load_pattern=os.path.join(gold_folder,
                                                             const.EXT_GOLDSTD),
                                   load_func=utils._imread_goldstd)

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

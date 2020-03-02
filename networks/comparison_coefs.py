import constants as const
import utils
from skimage import io

import os


NETWORK = 'tiramisu'  # available: 'unet', 'tiramisu'
TIRAMISU_MODEL = 'tiramisu-67'  # available: 'tiramisu-56', 'tiramisu-67'
TIRAMISU_LAYERS = TIRAMISU_MODEL[8:]
NETWORK_FOLDER = {'tiramisu': const.FOLDER_PRED_TIRAMISU + TIRAMISU_LAYERS,
                  'unet': const.FOLDER_PRED_UNET,
                  'tiramisu_3d': const.FOLDER_PRED_TIRAMISU_3D + TIRAMISU_LAYERS,
                  'unet_3d': const.FOLDER_PRED_UNET_3D}

if NETWORK in ('tiramisu', 'tiramisu_3d'):
    PATH_COEFFICIENTS = os.path.join(const.FOLDER_COMP_COEF,
                                     NETWORK + TIRAMISU_LAYERS)
else:
    PATH_COEFFICIENTS = os.path.join(const.FOLDER_COMP_COEF,
                                     NETWORK)

if not os.path.isdir(PATH_COEFFICIENTS):
    os.makedirs(PATH_COEFFICIENTS)


def read_data(sample, folder_prediction=NETWORK_FOLDER[NETWORK],
              is_registered=False):
    """
    """
    if is_registered:
        folder_pred = os.path.join(folder_prediction,
                                   sample['folder'] + '_REG',
                                   const.SUBFOLDER_PRED,
                                   '*' + const.EXT_PRED)
        folder_goldstd = os.path.join(const.FOLDER_GOLDSTD,
                                      sample['folder'],
                                      const.SUBFOLDER_GOLDSTD_REG,
                                      '*' + const.EXT_GOLDSTD)
    else:
        folder_pred = os.path.join(folder_prediction,
                                   sample['folder'],
                                   const.SUBFOLDER_PRED,
                                   '*' + const.EXT_PRED)
        folder_goldstd = os.path.join(const.FOLDER_GOLDSTD,
                                      sample['folder'],
                                      const.SUBFOLDER_GOLDSTD,
                                      '*' + const.EXT_GOLDSTD)

    data_prediction = io.ImageCollection(load_pattern=folder_pred,
                                         load_func=utils.imread_prediction)
    data_goldstd = io.ImageCollection(load_pattern=folder_goldstd,
                                      load_func=utils.imread_goldstd)
    return data_prediction, data_goldstd


for sample in const.SAMPLES_BUNCH2:
    if sample['has_goldstd']:
        print(f"Now processing {sample['folder']}.")
        if sample['registered_path']:
            data_prediction, data_goldstd = read_data(sample, is_registered=True)
        else:
            data_prediction, data_goldstd = read_data(sample, is_registered=False)

        data_prediction = data_prediction[slice(*sample['segmentation_interval'])]

        # coefficients will receive the folder name as a filename.
        filename = f"{PATH_COEFFICIENTS}/{sample['folder']}-{NETWORK}_coefs.csv"

        _, _ = utils.measure_all_coefficients(data_prediction,
                                              data_goldstd,
                                              save_coef=True,
                                              filename=filename)

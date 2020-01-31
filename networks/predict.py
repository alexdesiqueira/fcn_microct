from skimage import io

import auxiliar
import constants as const
import os
import tensorflow as tf


NETWORK = 'unet'
SAMPLES = const.SAMPLES_BUNCH2
WEIGHTS = 'larson_unet.hdf5'


for sample in SAMPLES:
    print(f"# Now reading sample {sample['path']}.")
    pattern = os.path.join(sample['path'], '*' + const.EXT_SAMPLE)
    data_sample = io.ImageCollection(load_pattern=pattern)

    print('# Processing...')
    folder = auxiliar._aux_prediction_folder(network=NETWORK)
    auxiliar._aux_process_sample(folder, data=data_sample, weights=WEIGHTS)

    tf.keras.backend.clear_session()  # resetting session state

    if sample['registered_path']:
        pattern = os.path.join(sample['registered_path'],
                               '*' + const.EXT_REG)
        data_sample = io.ImageCollection(load_pattern=pattern)

        print('# Processing registered sample...')
        folder = auxiliar._aux_prediction_folder(network=NETWORK)
        folder_reg = folder + '_REG'
        auxiliar._aux_process_sample(folder_reg,
                                     data=data_sample,
                                     weights=WEIGHTS)

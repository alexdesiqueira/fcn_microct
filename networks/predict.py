from skimage import io

import auxiliar
import constants as const
import os


NETWORK = 'tiramisu'
SAMPLES = const.SAMPLES_BUNCH2
WEIGHTS = '../coefficients/larson2019_tiramisu-67/larson_tiramisu-67.hdf5'


for sample in SAMPLES:
    print(f"# Now reading sample {sample['path']}.")
    pattern = os.path.join(sample['path'], '*' + const.EXT_SAMPLE)
    data_sample = io.ImageCollection(load_pattern=pattern)

    print('# Processing...')
    folder = os.path.join(auxiliar._aux_prediction_folder(network=NETWORK),
                          sample['folder'])
    auxiliar._aux_process_sample(folder,
                                 data=data_sample,
                                 weights=WEIGHTS,
                                 network=NETWORK)

    if sample['registered_path']:
        pattern = os.path.join(sample['registered_path'],
                               '*' + const.EXT_REG)
        data_sample = io.ImageCollection(load_pattern=pattern)

        print('# Processing registered sample...')
        folder = os.path.join(auxiliar._aux_prediction_folder(network=NETWORK),
                              sample['folder'] + '_REG')
        auxiliar._aux_process_sample(folder,
                                     data=data_sample,
                                     weights=WEIGHTS,
                                     network=NETWORK)

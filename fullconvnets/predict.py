from skimage import io

import constants as const
import os
import utils


NETWORK = 'unet_3d'
SAMPLES = const.SAMPLES_BUNCH2
WEIGHTS = '../coefficients/larson2019_unet_3d/larson_unet_3d.hdf5'


for sample in SAMPLES:
    print(f"# Now reading sample {sample['path']}.")
    pattern = os.path.join(sample['path'], '*' + const.EXT_SAMPLE)
    data_sample = io.ImageCollection(load_pattern=pattern)

    print('# Processing...')
    folder = os.path.join(utils.prediction_folder(network=NETWORK),
                          sample['folder'])
    utils.process_sample(folder,
                         data=data_sample,
                         weights=WEIGHTS,
                         network=NETWORK)

    if sample['registered_path']:
        pattern = os.path.join(sample['registered_path'],
                               '*' + const.EXT_REG)
        data_sample = io.ImageCollection(load_pattern=pattern)

        print('# Processing registered sample...')
        folder = os.path.join(utils.prediction_folder(network=NETWORK),
                              sample['folder'] + '_REG')
        utils.process_sample(folder,
                             data=data_sample,
                             weights=WEIGHTS,
                             network=NETWORK)

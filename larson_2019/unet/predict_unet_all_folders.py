from model import unet
from skimage import io, util

import os
import tensorflow as tf
import utils

print('# Preparing folders...')
base_folder = '/home/alex/data/larson_2019'

for test_path, subfolders, _ in [os.walk(os.path.join(base_folder, 'Recons/Bunch2WoPR/wet/')),
                                 os.walk(os.path.join(base_folder, 'Recons/Bunch2WoPR/cured/'))]:
    if not subfolders:
        folder = test_path.split('/')[-1]
        pred_path = os.path.join(base_folder, 'data', folder, 'predict')
        over_path = os.path.join(base_folder, 'data', folder, 'overlap')

        for aux in [pred_path, over_path]:
            if not os.path.isdir(aux):
                os.makedirs(aux)

        print('# Reading images...')
        test_images = io.ImageCollection(os.path.join(test_path, '*.tiff'))

        print('# Processing...')
        for idx, image in enumerate(test_images):
            prediction = utils.predict_on_image(image, weights='576x576_withpad/take_final/larson_unet.hdf5')
            overlap = utils.overlap_predictions(image, prediction)
            fname = '%04d.png' % (idx)
            io.imsave(os.path.join(pred_path, fname), util.img_as_ubyte(prediction))
            io.imsave(os.path.join(over_path, fname), util.img_as_ubyte(overlap))

            tf.keras.backend.clear_session()  # For easy reset of notebook state.

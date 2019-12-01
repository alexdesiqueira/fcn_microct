from model import unet
from skimage import io

import os
import utils

import tensorflow as tf


print('# Preparing folders...')
base_folder = '/home/alex/data/larson_2019/data/'
test_path = os.path.join(base_folder, 'test')
pred_path = os.path.join(test_path, 'predict')
over_path = os.path.join(test_path, 'overlap')

for folder in [pred_path, over_path]:
    if not os.path.isdir(folder):
        os.makedirs(folder)

print('# Reading images...')
test_images = io.ImageCollection(os.path.join(test_path, 'image/*.png'))

print('# Processing...')
for idx, image in enumerate(test_images):
    prediction = utils.predict_on_image(image, weights='576x576_withpad/take_3/larson_unet-576x576_withpad-take3.hdf5')
    overlap = utils.overlap_predictions(image, prediction)
    fname = '%04d.png' % (idx)
    io.imsave(os.path.join(pred_path, fname), prediction)
    io.imsave(os.path.join(over_path, fname), overlap)

    tf.keras.backend.clear_session()  # For easy reset of notebook state.

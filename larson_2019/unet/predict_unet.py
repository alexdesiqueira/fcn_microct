from model import unet
from os.path import join
from skimage import io

import utils


base_folder = '/home/alex/data/larson_2019/data/'
test_path = join(base_folder, 'test/image')

test_images = io.ImageCollection(join(test_path, '*.png'))
for image in test_images:
    result = utils.predict_on_image(image)

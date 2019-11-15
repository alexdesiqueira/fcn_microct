from model import unet
from os.path import join

import data
import utils


base_folder = '../data/larson_2019/data'

test_path = join(base_folder, 'test/image/crop')
test_gene = data.test_generator(test_path, num_image=100)
model = unet(input_size=(512, 512, 1))
model.load_weights('unet_larson.hdf5')
results = model.predict_generator(test_gene, 100, verbose=1)
utils.save_predictions(join(base_folder, 'test/image/predict'), results)

from model import unet
from os.path import join

import data
import utils


base_folder = '/home/alex/data/larson_2019/data'

test_path = join(base_folder, 'test/image')
test_gene = data.test_generator(test_path,
                                target_size=(576, 576),
                                pad_width=0)

model = unet(input_size=(576, 576, 1))
model.load_weights('larson_unet.hdf5')
results = model.predict(test_gene, steps=1000, verbose=1)
utils.save_predictions(join(base_folder, 'test/predict'), results)

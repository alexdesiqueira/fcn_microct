import os


# defining folder structure.
# base folder
FOLDER_BASE = '/home/alex/data/larson_2019'

# Larson's samples
FOLDER_SAMPLE = os.path.join(FOLDER_BASE, 'Recons/Bunch2WoPR/')
EXT_SAMPLE = '.tiff'

# gold standard and registration (Larson's et al) folders
# FOLDER_GOLDSTD can be used as the alternative sample folder
# — for the registered stacks — as well.
FOLDER_GOLDSTD = os.path.join(FOLDER_BASE, 'Seg/Bunch2/')
SUBFOLDER_GOLDSTD = '19_Gray_image'
SUBFOLDER_GOLDSTD_REG = 'Registered/19_Gray_image'
SUBFOLDER_REG = 'Registered/Bunch2WoPR'
EXT_GOLDSTD = '.tif'

# setting image constants.
PAD_WIDTH = 16
ROWS, COLS = (256, 256)
STEP = 256
WINDOW_SHAPE = (ROWS + 2*PAD_WIDTH, COLS + 2*PAD_WIDTH)

# the training folders and its subfolders.
FOLDER_TRAINING_ORIG = os.path.join(FOLDER_BASE, 'data_training/original')
FOLDER_TRAIN_IMAGE_ORIG = os.path.join(FOLDER_TRAINING_ORIG, 'train/image/')
FOLDER_TRAIN_LABEL_ORIG = os.path.join(FOLDER_TRAINING_ORIG, 'train/label/')
FOLDER_VAL_IMAGE_ORIG = os.path.join(FOLDER_TRAINING_ORIG, 'validate/image/')
FOLDER_VAL_LABEL_ORIG = os.path.join(FOLDER_TRAINING_ORIG, 'validate/label/')

FOLDER_TRAINING_CROP = os.path.join(FOLDER_BASE, 'data_training/cropped')
FOLDER_TRAIN_IMAGE_CROP = os.path.join(FOLDER_TRAINING_CROP, 'train/image/')
FOLDER_TRAIN_LABEL_CROP = os.path.join(FOLDER_TRAINING_CROP, 'train/label/')
FOLDER_VAL_IMAGE_CROP = os.path.join(FOLDER_TRAINING_CROP, 'validate/image/')
FOLDER_VAL_LABEL_CROP = os.path.join(FOLDER_TRAINING_CROP, 'validate/label/')

# interval images used in the training.
INTERVAL_TRAIN_CURED = [160, 510]
INTERVAL_TRAIN_WET = [510, 860]
INTERVAL_VAL_CURED = [860, 1010]
INTERVAL_VAL_WET = [1010, 1160]

# our results.
FOLDER_RESULTS_UNET = os.path.join(FOLDER_BASE, 'data/unet')
FOLDER_RESULTS_TIRAMISU = os.path.join(FOLDER_BASE, 'data/tiramisu')
FOLDER_RESULTS_3DUNET = os.path.join(FOLDER_BASE, 'data/unet_3d')
SUBFOLDER_PRED = 'predict'
SUBFOLDER_OVER = 'overlap'
EXT_PRED = '.png'
EXT_OVER = '.png'

# defining sample folder names.
FOLDER_232p1_wet = 'rec20160324_055424_232p1_wet_1cm_cont_4097im_1500ms_17keV_13_a.h5'
FOLDER_232p3_cured = 'rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5'
FOLDER_232p3_wet = 'rec20160318_191511_232p3_2cm_cont__4097im_1500ms_ML17keV_6.h5'
FOLDER_235p1_wet = 'rec20160324_123639_235p1_wet_0p7cm_cont_4097im_1500ms_17keV_14.h5'
FOLDER_235p4_cured = 'rec20160327_003824_235p4_cured_1p15cm_cont_4097im_1500ex_17keV_22.h5'
FOLDER_235p4_wet = 'rec20160326_175540_235p4_wet_1p15cm_cont_4097im_1500ex_17keV_20.h5'
FOLDER_244p1_cured = 'rec20160320_160251_244p1_1p5cm_cont_4097im_1500ms_ML17keV_9.h5'
FOLDER_244p1_wet = 'rec20160318_223946_244p1_1p5cm_cont__4097im_1500ms_ML17keV_7.h5'
FOLDER_245p1_wet = 'rec20160327_160624_245p1_wet_1cm_cont_4097im_1500ex_17keV_23.h5'

"""
* Defining Bunch2 samples

Samples have their own folder and its path; they can have a gold standard
(Larson's et al solution), its path and a corresponding segmentation interval.
We are interested also in some registered data, available for two of the cured
samples with gold standards. Therefore we define a registered path as well.
"""
# [TODO: CHECK IF FOLDERS MATCH THE SAMPLES!]

SAMPLE_232p1_wet = {
    'folder': FOLDER_232p1_wet,
    'path': os.path.join(FOLDER_SAMPLE,
                         FOLDER_232p1_wet),
    'has_goldstd': True,
    'path_goldstd': os.path.join(FOLDER_GOLDSTD,
                                 FOLDER_232p1_wet,
                                 SUBFOLDER_GOLDSTD),
    'segmentation_interval': [160, 1160],
    'registered_path': None
}

SAMPLE_232p3_cured = {
    'folder': FOLDER_232p3_cured,
    'path': os.path.join(FOLDER_SAMPLE,
                         FOLDER_232p1_wet),
    'has_goldstd': True,
    'path_goldstd': os.path.join(FOLDER_GOLDSTD,
                                 FOLDER_232p3_cured,
                                 SUBFOLDER_GOLDSTD_REG),
    'segmentation_interval': [159, 1159],  # [0, 1000],
    'registered_path': os.path.join(FOLDER_GOLDSTD,
                                    FOLDER_232p3_cured,
                                    SUBFOLDER_REG)
}

SAMPLE_232p3_wet = {
    'folder': FOLDER_232p3_wet,
    'path': os.path.join(FOLDER_SAMPLE,
                         FOLDER_232p3_wet),
    'has_goldstd': True,
    'path_goldstd': os.path.join(FOLDER_GOLDSTD,
                                 FOLDER_232p3_wet,
                                 SUBFOLDER_GOLDSTD),
    'segmentation_interval': [159, 1159],  # [0, 1000],
    'registered_path': None
}

SAMPLE_235p1_wet = {
    'folder': FOLDER_235p1_wet,
    'path': os.path.join(FOLDER_SAMPLE,
                         FOLDER_235p1_wet),
    'has_goldstd': False,
    'path_goldstd': None,
    'segmentation_interval': None,
    'registered_path': None
}

SAMPLE_235p4_cured = {
    'folder': FOLDER_235p4_cured,
    'path': os.path.join(FOLDER_SAMPLE,
                         FOLDER_235p4_cured),
    'has_goldstd': False,
    'path_goldstd': None,
    'segmentation_interval': None,
    'registered_path': os.path.join(FOLDER_GOLDSTD,
                                    FOLDER_235p4_cured,
                                    SUBFOLDER_REG)
}

SAMPLE_235p4_wet = {
    'folder': FOLDER_235p4_wet,
    'path': os.path.join(FOLDER_SAMPLE,
                         FOLDER_235p4_wet),
    'has_goldstd': False,
    'path_goldstd': None,
    'segmentation_interval': None,
    'registered_path': None
}

SAMPLE_244p1_cured = {
    'folder': FOLDER_244p1_cured,
    'path': os.path.join(FOLDER_SAMPLE,
                         FOLDER_244p1_cured),
    'has_goldstd': True,
    'path_goldstd': os.path.join(FOLDER_GOLDSTD,
                                 FOLDER_244p1_cured,
                                 SUBFOLDER_GOLDSTD_REG),
    'segmentation_interval': [150, 1150],
    'registered_path': os.path.join(FOLDER_GOLDSTD,
                                    FOLDER_235p4_cured,
                                    SUBFOLDER_REG)
}

SAMPLE_244p1_wet = {
    'folder': FOLDER_244p1_wet,
    'path': os.path.join(FOLDER_SAMPLE,
                         FOLDER_244p1_wet),
    'has_goldstd': True,
    'path_goldstd': os.path.join(FOLDER_GOLDSTD,
                                 FOLDER_244p1_wet,
                                 SUBFOLDER_GOLDSTD),
    'segmentation_interval': [150, 1150],
    'registered_path': None
}

SAMPLE_245p1_wet = {
    'folder': FOLDER_245p1_wet,
    'path': os.path.join(FOLDER_SAMPLE,
                         FOLDER_245p1_wet),
    'has_goldstd': False,
    'path_goldstd': None,
    'segmentation_interval': None,
    'registered_path': None
}

# putting all samples in a list, to ease looping through them.
SAMPLES_BUNCH2 = [SAMPLE_232p1_wet, SAMPLE_232p3_cured, SAMPLE_232p3_wet,
                  SAMPLE_235p1_wet, SAMPLE_235p4_cured, SAMPLE_235p4_wet,
                  SAMPLE_244p1_cured, SAMPLE_244p1_wet, SAMPLE_245p1_wet]

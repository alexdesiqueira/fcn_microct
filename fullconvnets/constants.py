import numpy as np
import os


# available networks.
AVAILABLE_2D_NETS = ('tiramisu', 'unet')
AVAILABLE_3D_NETS = ('tiramisu_3d', 'unet_3d')

# available Tiramisu models.
AVAILABLE_TIRAMISU_MODELS = ('tiramisu-56', 'tiramisu-67')

# defining a color dict with each class.
CLASS_0 = [0, 0, 0]
CLASS_1 = [128, 128, 128]
CLASS_2 = [128, 0, 0]
CLASS_3 = [192, 192, 128]
CLASS_4 = [128, 64, 128]
CLASS_5 = [60, 40, 222]
CLASS_6 = [128, 128, 0]
CLASS_7 = [192, 128, 128]
CLASS_8 = [64, 64, 128]
CLASS_9 = [64, 0, 128]

COLOR_DICT = np.array([CLASS_0, CLASS_1, CLASS_2, CLASS_3, CLASS_4,
                       CLASS_5, CLASS_6, CLASS_7, CLASS_8, CLASS_9])

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
EXT_GOLDSTD = '.tif'
SUBFOLDER_REG = 'Registered/Bunch2WoPR'
SUBFOLDER_GOLDSTD_REG = 'Registered/19_Gray_image'
EXT_REG = '.tif'

# setting image constants.
PAD_WIDTH = 16
ROWS, COLS = (256, 256)
STEP = 256
WINDOW_SHAPE = (ROWS + 2*PAD_WIDTH, COLS + 2*PAD_WIDTH)

PAD_WIDTH_3D = 16
PLANES_3D, ROWS_3D, COLS_3D = (32, 32, 32)
STEP_3D = 32
WINDOW_SHAPE_3D = (PLANES_3D + 2*PAD_WIDTH_3D,
                   ROWS_3D + 2*PAD_WIDTH_3D,
                   COLS_3D + 2*PAD_WIDTH_3D)

# the main subfolders, and extensions.
SUBFOLDER_DATA_TRAIN = 'data_training'
SUBFOLDER_DATA_PRED = 'data_prediction'
SUBFOLDER_TRAIN = 'train'
SUBFOLDER_VALIDATE = 'validate'
SUBFOLDER_IMAGE = 'image'
SUBFOLDER_LABEL = 'label'
SUBFOLDER_PRED = 'predict'
SUBFOLDER_OVER = 'overlap'
EXT_PRED = '.png'
EXT_OVER = '.png'

# the training folders and its subfolders.
FOLDER_TRAINING_ORIG = os.path.join(FOLDER_BASE,
                                    SUBFOLDER_DATA_TRAIN,
                                    'original')
FOLDER_TRAIN_IMAGE_ORIG = os.path.join(FOLDER_TRAINING_ORIG,
                                       SUBFOLDER_TRAIN,
                                       SUBFOLDER_IMAGE)
FOLDER_TRAIN_LABEL_ORIG = os.path.join(FOLDER_TRAINING_ORIG,
                                       SUBFOLDER_TRAIN,
                                       SUBFOLDER_LABEL)
FOLDER_VAL_IMAGE_ORIG = os.path.join(FOLDER_TRAINING_ORIG,
                                     SUBFOLDER_VALIDATE,
                                     SUBFOLDER_IMAGE)
FOLDER_VAL_LABEL_ORIG = os.path.join(FOLDER_TRAINING_ORIG,
                                     SUBFOLDER_VALIDATE,
                                     SUBFOLDER_LABEL)

FOLDER_TRAINING_CROP = os.path.join(FOLDER_BASE,
                                    SUBFOLDER_TRAIN,
                                    'cropped')
FOLDER_TRAIN_IMAGE_CROP = os.path.join(FOLDER_TRAINING_CROP,
                                       SUBFOLDER_TRAIN,
                                       SUBFOLDER_IMAGE)
FOLDER_TRAIN_LABEL_CROP = os.path.join(FOLDER_TRAINING_CROP,
                                       SUBFOLDER_TRAIN,
                                       SUBFOLDER_LABEL)
FOLDER_VAL_IMAGE_CROP = os.path.join(FOLDER_TRAINING_CROP,
                                     SUBFOLDER_VALIDATE,
                                     SUBFOLDER_IMAGE)
FOLDER_VAL_LABEL_CROP = os.path.join(FOLDER_TRAINING_CROP,
                                     SUBFOLDER_VALIDATE,
                                     SUBFOLDER_LABEL)

FOLDER_TRAINING_CROP_3D = os.path.join(FOLDER_BASE,
                                       SUBFOLDER_TRAIN,
                                       'cropped_3d')
FOLDER_TRAIN_IMAGE_CROP_3D = os.path.join(FOLDER_TRAINING_CROP_3D,
                                          SUBFOLDER_TRAIN,
                                          SUBFOLDER_IMAGE)
FOLDER_TRAIN_LABEL_CROP_3D = os.path.join(FOLDER_TRAINING_CROP_3D,
                                          SUBFOLDER_TRAIN,
                                          SUBFOLDER_LABEL)
FOLDER_VAL_IMAGE_CROP_3D = os.path.join(FOLDER_TRAINING_CROP_3D,
                                        SUBFOLDER_VALIDATE,
                                        SUBFOLDER_IMAGE)
FOLDER_VAL_LABEL_CROP_3D = os.path.join(FOLDER_TRAINING_CROP_3D,
                                        SUBFOLDER_VALIDATE,
                                        SUBFOLDER_LABEL)

# interval images used in the training.
INTERVAL_TRAIN_CURED = [160, 510]
INTERVAL_TRAIN_WET = [510, 860]  # a total of 700 training images
INTERVAL_VAL_CURED = [860, 1010]
INTERVAL_VAL_WET = [1010, 1158]  # a total of 298 validation images

# our prediction folders.
FOLDER_PRED_TIRAMISU = os.path.join(FOLDER_BASE,
                                    SUBFOLDER_DATA_PRED,
                                    'tiramisu')
FOLDER_PRED_TIRAMISU_3D = os.path.join(FOLDER_BASE,
                                       SUBFOLDER_DATA_PRED,
                                       'tiramisu_3d')
FOLDER_PRED_UNET = os.path.join(FOLDER_BASE,
                                SUBFOLDER_DATA_PRED,
                                'unet')
FOLDER_PRED_UNET_3D = os.path.join(FOLDER_BASE,
                                   SUBFOLDER_DATA_PRED,
                                   'unet_3d')

# setting the folder to store comparison coefficients.
FOLDER_COMP_COEF = os.path.join(FOLDER_BASE, 'comp_coefficients')

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
    'segmentation_interval': [159, 1159],  # [0, 1000]
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
    'segmentation_interval': [0, 1000],
    # was [159, 1159], but we have only 1000 slices in the sample
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
                                    FOLDER_244p1_cured,
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

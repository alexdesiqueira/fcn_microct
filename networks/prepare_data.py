import constants as const
import os
import shutil


def copy_training_samples():
    """
    """
    # checking if folders exist.
    for folder in [const.FOLDER_TRAIN_IMAGE,
                   const.FOLDER_TRAIN_LABEL,
                   const.FOLDER_VAL_IMAGE,
                   const.FOLDER_VAL_LABEL]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    # getting image intervals.
    start, end = const.INTERVAL_TRAIN_CURED
    for number in range(start, end):
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['registered_path']}/Reg_0{number}.tif",
                    dst=f"{const.FOLDER_TRAIN_IMAGE}")
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['path_goldstd']}/19_Gray_0{number}.tif",
                    dst=f"{const.FOLDER_TRAIN_LABEL}")

    start, end = const.INTERVAL_VAL_CURED
    for number in range(start, end):
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['registered_path']}/Reg_0{number}.tif",
                    dst=f"{const.FOLDER_VAL_IMAGE}")
        shutil.copy(src=f"{const.SAMPLE_232p3_cured['path_goldstd']}/19_Gray_0{number}.tif",
                    dst=f"{const.FOLDER_VAL_LABEL}")

    start, end = const.INTERVAL_TRAIN_WET
    for number in range(start, end):
        if number < 1000:
            aux_image = 'rec_SFRR_2600_B0p2_00'
            aux_label = '19_Gray_0'
        else:
            aux_image = 'rec_SFRR_2600_B0p2_0'
            aux_label = '19_Gray_'
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path']}/{aux_image}{number}.tiff",
                    dst=f"{const.FOLDER_TRAIN_IMAGE}")
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path_goldstd']}/{aux_label}{number}.tif",
                    dst=f"{const.FOLDER_TRAIN_LABEL}")

    start, end = const.INTERVAL_VAL_WET
    for number in range(start, end):
        if number < 1000:
            aux_image = 'rec_SFRR_2600_B0p2_00'
            aux_label = '19_Gray_0'
        else:
            aux_image = 'rec_SFRR_2600_B0p2_0'
            aux_label = '19_Gray_'
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path']}/{aux_image}{number}.tiff",
                    dst=f"{const.FOLDER_VAL_IMAGE}")
        shutil.copy(src=f"{const.SAMPLE_232p3_wet['path_goldstd']}/{aux_label}{number}.tif",
                    dst=f"{const.FOLDER_VAL_LABEL}")

    return None

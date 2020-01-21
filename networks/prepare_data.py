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
    for interval in [const.INTERVAL_TRAIN_CURED,
                     const.INTERVAL_VAL_CURED]:
        start, end = interval

        for number in range(start, end):
            shutil.copy(src=f"{const.SAMPLE_232p3_cured['registered_path']}/Reg_0{number}.tif",
                        dst=f"{const.FOLDER_TRAIN_IMAGE}")
            shutil.copy(src=f"const.SAMPLE_232p3_cured['path_goldstd']/19_Gray_0{number}.tif",
                        dst=f"{const.FOLDER_TRAIN_LABEL}")

    for interval in [const.INTERVAL_TRAIN_WET,
                     const.INTERVAL_VAL_WET]:
    # getting sample paths.

    wet_path = const.SAMPLE_232p3_wet['path']



    return None

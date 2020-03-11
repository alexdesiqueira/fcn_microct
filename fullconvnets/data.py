from skimage import io, transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import generator_3d
import numpy as np
import os


def train_generator(batch_size, train_path, subfolders, augmentation_dict,
                    color_mode=('grayscale', 'grayscale'), n_classes=2,
                    target_size=(256, 256), save_to_dir=None,
                    save_prefix=('image', 'label'), seed=1,
                    multichannel=False):
    """A generator to train fully convolutional networks.

    Parameters
    ----------
    batch_size : int
        The batch size returned by the generator.
    train_path : str
        The system path for the training images.
    subfolders : list
        Names of image and label subfolders, stored in the train_path.
    augmentation_dict : dict
        Dictionary with augmentation arguments passed to the image or chunk
        data generator.
    color_mode : list (default : ('grayscale', 'grayscale'))
        Names of color modes for image and label.
    n_classes : int (default : 2)
        Number of classes contained in labels.
    target_size : array_like (default : (256, 256))
        Size of the desired training windows.
    save_to_dir : str or None (default : None)
        If not None, the system path where the augmented images and labels
        will be stored.
    save_prefix : str (default : ('image', 'label'))
        names of image and label subfolders, used when save_to_dir is not
        None.
    seed : int (default : 1)
        Seed used to generate augmentations on image and label.
    multichannel : bool (default : False)
        False if the input images have only one channel.

    Yields
    ------
    (image, label) : list
        List with an augmented pair of image and label.

    Notes
    -----
    train_generator generates image and label, setting the same seed
    for both.

    subfolders can contain a list of lists, pointing to more than one
    subfolder. For example, we can point to two image subfolders 'image1' and
    'image2' with subfolder=(('image1', 'image2'), 'label').

    If target_size has two or three dimensions, aug_dict will be passed to
    TensorFlow's keras.preprocessing.image.ImageDataGenerator or
    generator_3d.ChunkDataGenerator, respectively.

    References
    ----------
    .. [1]  https://github.com/zhixuhao/unet

    Examples
    --------
    >>> train_gen = data.train_generator(batch_size=2,
                                         train_path='train',
                                         subfolders=('image', 'label'),
                                         augmentation_dict=dict(rotation_range=0.1,
                                                                horizontal_flip=True,
                                                                vertical_flip=True),
                                         target_size=(128, 128),
                                         save_to_dir=None)
    """
    if len(target_size) == 2:
        image_datagen = ImageDataGenerator(**augmentation_dict)
        label_datagen = ImageDataGenerator(**augmentation_dict)
    elif len(target_size) == 3:
        image_datagen = generator_3d.ChunkDataGenerator(**augmentation_dict)
        label_datagen = generator_3d.ChunkDataGenerator(**augmentation_dict)

    # creating a generator for the input images.
    image_gen = image_datagen.flow_from_directory(
        train_path,
        classes=[subfolders[0]],
        class_mode=None,
        color_mode=color_mode[0],
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix[0],
        seed=seed)

    # creating a generator for the input labels.
    label_gen = label_datagen.flow_from_directory(
        train_path,
        classes=[subfolders[1]],
        class_mode=None,
        color_mode=color_mode[1],
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix[1],
        seed=seed)

    train_gen = zip(image_gen, label_gen)

    for (image, label) in train_gen:
        image, label = _adjust_data(image, label, n_classes, multichannel)
        yield (image, label)


def _adjust_data(image, label, n_classes=2, multichannel=False):
    """Helps set data according to its channels, classes, and labels."""
    if multichannel:
        image = image / 255
        label = label[..., 0]
        aux_label = np.zeros(label.shape + (n_classes,))

        for num in range(n_classes):
            aux_label[label == num, num] = 1

        if multichannel:
            batch, rows, cols, classes = aux_label.shape
            aux_label = np.reshape(aux_label,
                                   (batch,
                                    rows*cols,
                                    classes))
        else:
            rows, cols, batch = aux_label.shape
            aux_label = np.reshape(aux_label, (rows*cols, batch))
        label = aux_label

    elif image.max() > 1:
        image = image / 255
        label = label / 255
        label[label > 0.5] = 1
        label[label <= 0.5] = 0

    return (image, label)

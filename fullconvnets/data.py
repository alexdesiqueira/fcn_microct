from skimage import io, transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import generator_3d
import numpy as np
import os


def adjust_data(image, labels, num_class=2, multichannel=False):
    if multichannel:
        image = image / 255
        labels = labels[..., 0]
        aux_labels = np.zeros(labels.shape + (num_class,))

        for num in range(num_class):
            aux_labels[labels == num, num] = 1

        if multichannel:
            batch, rows, cols, classes = aux_labels.shape
            aux_labels = np.reshape(aux_labels,
                                    (batch,
                                     rows*cols,
                                     classes))
        else:
            rows, cols, batch = aux_labels.shape
            aux_labels = np.reshape(aux_labels, (rows*cols, batch))
        labels = aux_labels

    elif image.max() > 1:  # terrible, but I have no options anymore
        image = image / 255
        labels = labels / 255
        labels[labels > 0.5] = 1
        labels[labels <= 0.5] = 0

    return (image, labels)


def test_generator(test_path, target_size=(256, 256), pad_width=16,
                   multichannel=False):
    """
    """
    images = io.ImageCollection(os.path.join(test_path, '*.png'))
    for image in images:
        image = image / 255
        if image.shape != target_size:
            image = transform.resize(image, target_size)
        # padding image to correct slicing after
        image = np.pad(image, pad_width=pad_width, mode='reflect')
        if not multichannel:
            image = np.reshape(image, image.shape+(1,))
        image = np.reshape(image, (1,)+image.shape)

        yield image


def train_generator(batch_size, train_path, image_folder, label_folder,
                    aug_dict, image_color_mode='grayscale',
                    label_color_mode='grayscale', image_save_prefix='image',
                    label_save_prefix='label', multichannel=False,
                    num_class=2, save_to_dir=None, target_size=(256, 256),
                    seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and label_datagen
    to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set
    save_to_dir = "your path"
    '''
    if len(target_size) == 2:
        image_datagen = ImageDataGenerator(**aug_dict)
        label_datagen = ImageDataGenerator(**aug_dict)
    elif len(target_size) == 3:
        image_datagen = generator_3d.ChunkDataGenerator(**aug_dict)
        label_datagen = generator_3d.ChunkDataGenerator(**aug_dict)

    image_gen = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    label_gen = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed)

    train_gen = zip(image_gen, label_gen)

    for (image, label) in train_gen:
        image, label = adjust_data(image, label, num_class, multichannel)
        yield (image, label)
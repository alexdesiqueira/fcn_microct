from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io, transform

import numpy as np
import os


def adjust_data(image, labels, flag_multi_class, num_class):
    if flag_multi_class:
        image = image / 255
        if len(labels.shape) == 4:
            labels = labels[:, :, :, 0]
        else:
            labels = labels[:, :, 0]
        aux_labels = np.zeros(labels.shape + (num_class,))

        for num in range(num_class):
            aux_labels[labels == num, num] = 1

        if flag_multi_class:
            depth, rows, cols, classes = aux_labels.shape
            aux_labels = np.reshape(aux_labels, (depth, rows*cols, classes))
        else:
            rows, cols, classes = aux_labels.shape
            aux_labels = np.reshape(aux_labels, (rows*cols, classes))
        labels = aux_labels

    elif np.max(image) > 1:
        image = image / 255
        labels = labels / 255
        labels[labels > 0.5] = 1
        labels[labels <= 0.5] = 0

    return (image, labels)


def test_generator(test_path, target_size=(256, 256), pad_width=16,
                   flag_multi_class=False, as_gray=True):
    images = io.ImageCollection(os.path.join(test_path, '*.png'))
    for image in images:
        image = image / 255
        if image.shape != target_size:
            image = transform.resize(image, target_size)
        # padding image
        image = np.pad(image, pad_width=pad_width, mode='reflect')
        if not flag_multi_class:
            image = np.reshape(image, image.shape+(1,))
        image = np.reshape(image, (1,)+image.shape)
        yield image


def train_generator(batch_size, train_path, image_folder, label_folder,
                    aug_dict, image_color_mode='grayscale',
                    label_color_mode='grayscale', image_save_prefix='image',
                    label_save_prefix='label', flag_multi_class=False,
                    num_class=2, save_to_dir=None, target_size=(256, 256),
                    seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and label_datagen
    to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set
    save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed)

    train_generator = zip(image_generator, label_generator)

    for (image, label) in train_generator:
        image, label = adjust_data(image, label, flag_multi_class, num_class)
        yield (image, label)
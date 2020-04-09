from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def unet(input_size=(256, 256, 1)):
    """Implements the two-dimensional version of the U-Net dense neural
    network.

    Parameters
    ----------
    input_size : (M, N, C) array-like, optional (default : (256, 256, 1))
        Shape of the input data in rows, columns, and channels.

    Returns
    -------
    model : model class
        A Keras model class.

    Notes
    -----
    U-Net is a fully convolutional network, aimed to applications on semantic
    segmentation.

    input_size does not contain the batch size. The default input,
    for example, indicates that the model expects images with 256 rows,
    256 columns, and one channel.

    This model is compiled with the optimizer Adam, according to _[1].

    References
    ----------
    .. [1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional
           Networks for Biomedical Image Segmentation,” in Medical Image
           Computing and Computer-Assisted Intervention – MICCAI 2015,
           Cham, 2015, pp. 234–241, doi: 10.1007/978-3-319-24574-4_28.
    .. [2] zhixuhao, "unet for image segmentation". Available at:
           https://github.com/zhixuhao/unet.

    Examples
    --------
    >>> from models import unet
    >>> model_unet = unet(input_size=(128, 128, 1))
    """
    n_classes = input_size[-1]
    inputs = layers.Input(input_size)

    # level 1 - down
    conv_down_1 = layers.Conv2D(filters=64,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(inputs)
    conv_down_1 = layers.Conv2D(filters=64,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(conv_down_1)
    max_pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(conv_down_1)

    # level 2 - down
    conv_down_2 = layers.Conv2D(filters=128,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(max_pool_1)
    conv_down_2 = layers.Conv2D(filters=128,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(conv_down_2)
    max_pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_down_2)

    # level 3 - down
    conv_down_3 = layers.Conv2D(filters=256,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(max_pool_2)
    conv_down_3 = layers.Conv2D(filters=256,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(conv_down_3)
    max_pool_3 = layers.MaxPooling2D(pool_size=(2, 2))(conv_down_3)

    # level 4 - down
    conv_down_4 = layers.Conv2D(filters=512,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(max_pool_3)
    conv_down_4 = layers.Conv2D(filters=512,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(conv_down_4)
    dropout_4 = layers.Dropout(0.5)(conv_down_4)
    max_pool_4 = layers.MaxPooling2D(pool_size=(2, 2))(dropout_4)

    # level 5 - down
    conv_down_5 = layers.Conv2D(filters=1024,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(max_pool_4)
    conv_down_5 = layers.Conv2D(filters=1024,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(conv_down_5)
    dropout_5 = layers.Dropout(0.5)(conv_down_5)

    # level 4 - up
    conv_up_4 = layers.Conv2D(filters=512,
                              kernel_size=2,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling2D(size=(2, 2))(dropout_5)
                              )
    merge_4 = layers.concatenate([dropout_4, conv_up_4], axis=3)
    conv_up_4 = layers.Conv2D(filters=512,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(merge_4)
    conv_up_4 = layers.Conv2D(filters=512,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(conv_up_4)

    # level 3 - up
    conv_up_3 = layers.Conv2D(filters=256,
                              kernel_size=2,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling2D(size=(2, 2))(conv_up_4)
                              )
    merge_3 = layers.concatenate([conv_down_3, conv_up_3], axis=3)
    conv_up_3 = layers.Conv2D(filters=256,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(merge_3)
    conv_up_3 = layers.Conv2D(filters=256,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(conv_up_3)

    # level 2 - up
    conv_up_2 = layers.Conv2D(filters=128,
                              kernel_size=2,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling2D(size=(2, 2))(conv_up_3)
                              )
    merge_2 = layers.concatenate([conv_down_2, conv_up_2], axis=3)
    conv_up_2 = layers.Conv2D(filters=128,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(merge_2)
    conv_up_2 = layers.Conv2D(filters=128,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(conv_up_2)

    # level 1 - up
    conv_up_1 = layers.Conv2D(filters=64,
                              kernel_size=2,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling2D(size=(2, 2))(conv_up_2)
                              )
    merge_1 = layers.concatenate([conv_down_1, conv_up_1], axis=3)
    conv_up_1 = layers.Conv2D(filters=64,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(merge_1)
    conv_up_1 = layers.Conv2D(filters=64,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(conv_up_1)

    # output segmentation map
    conv_up_1 = layers.Conv2D(filters=2,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(conv_up_1)

    # defining last convolution.
    if n_classes == 1:
        conv_output = layers.Conv2D(filters=n_classes,
                                    kernel_size=1,
                                    activation='sigmoid')(conv_up_1)
        loss = 'binary_crossentropy'
    else:
        conv_output = layers.Conv2D(filters=n_classes,
                                    kernel_size=1,
                                    activation='softmax')(conv_up_1)
        loss = 'categorical_crossentropy'

    model = Model(inputs, conv_output)
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss=loss,
                  metrics=['accuracy'])

    return model


def unet_3d(input_size=(64, 64, 64, 1)):
    """Implements the three-dimensional version of the U-Net dense neural
    network.

    Parameters
    ----------
    input_size : (P, M, N, C) array-like, optional (default : (64, 64, 64, 1))
        Shape of the input data in planes, rows, columns, and channels.

    Returns
    -------
    model : model class
        A Keras model class with its methods.

    Notes
    -----
    U-Net is a fully convolutional network, aimed to applications on semantic
    segmentation.

    The implementation follows Çiçek et al. _[1]:

    N -> 32 -> 64  ====================================================================>>  64+128 -> 64 -> 64 ~> N
                \                                                                           /
                64 -> 64 -> 128  ======================================>>  128+256 -> 128 -> 128
                              \                                              /
                              128 -> 128 -> 256  =======>>  256+512 -> 256 -> 256
                                              \               /
                                              256 -> 256 -> 512

    where:
    =>> : concat
    ~> : conv
    -> : conv + BN + ReLu
    \ : max pool
    / : up-conv

    input_size does not contain the batch size. The default input,
    for example, indicates that the model expects images with 256 rows,
    256 columns, and one channel.

    This model is compiled with the optimizer Adam.

    References
    ----------
    .. [1] Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O.
           Ronneberger, “3D U-Net: Learning Dense Volumetric Segmentation
           from Sparse Annotation,” arXiv:1606.06650 [cs], Jun. 2016.

    Examples
    --------
    >>> from models import unet_3d
    >>> model_3d_unet = unet_3d(input_size=(64, 64, 64, 1))
    """
    n_classes = input_size[-1]
    inputs = layers.Input(input_size)

    # level 1 - down
    conv_down_1 = layers.Conv3D(filters=32,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(inputs)
    conv_down_1 = layers.Conv3D(filters=64,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(conv_down_1)
    max_pool_1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_down_1)

    # level 2 - down
    conv_down_2 = layers.Conv3D(filters=64,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(max_pool_1)
    conv_down_2 = layers.Conv3D(filters=128,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(conv_down_2)
    max_pool_2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_down_2)

    # level 3 - down
    conv_down_3 = layers.Conv3D(filters=128,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(max_pool_2)
    conv_down_3 = layers.Conv3D(filters=256,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(conv_down_3)
    max_pool_3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_down_3)

    # level 4 - down
    conv_down_4 = layers.Conv3D(filters=256,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(max_pool_3)
    conv_down_4 = layers.Conv3D(filters=512,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                kernel_initializer='he_normal')(conv_down_4)

    # level 3 - up
    conv_up_3 = layers.Conv3D(filters=512,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling3D(size=(2, 2, 2))
                                  (conv_down_4)
                              )
    merge_3 = layers.concatenate([conv_down_3, conv_up_3], axis=-1)
    conv_up_3 = layers.Conv3D(filters=256,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(merge_3)
    conv_up_3 = layers.Conv3D(filters=256,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(conv_up_3)

    # level 2 - up
    conv_up_2 = layers.Conv3D(filters=256,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling3D(size=(2, 2, 2))
                                  (conv_up_3)
                              )
    merge_2 = layers.concatenate([conv_down_2, conv_up_2], axis=-1)
    conv_up_2 = layers.Conv3D(filters=128,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(merge_2)
    conv_up_2 = layers.Conv3D(filters=128,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(conv_up_2)

    # level 1 - up
    conv_up_1 = layers.Conv3D(filters=128,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling3D(size=(2, 2, 2))
                                  (conv_up_2)
                              )
    merge_1 = layers.concatenate([conv_down_1, conv_up_1], axis=-1)
    conv_up_1 = layers.Conv3D(filters=64,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(merge_1)
    conv_up_1 = layers.Conv3D(filters=64,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(conv_up_1)

    # defining last convolution.
    if n_classes == 1:
        # output segmentation map
        conv_up_1 = layers.Conv3D(filters=2,
                                  kernel_size=3,
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer='he_normal')(conv_up_1)
        conv_output = layers.Conv3D(filters=n_classes,
                                    kernel_size=1,
                                    activation='sigmoid')(conv_up_1)
        loss = 'binary_crossentropy'
    else:
        conv_output = layers.Conv3D(filters=n_classes,
                                    kernel_size=1,
                                    activation='softmax')(conv_up_1)
        loss = 'categorical_crossentropy'

    model = Model(inputs, conv_output)
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss=loss,
                  metrics=['accuracy'])

    return model

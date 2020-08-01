from tensorflow.keras import layers
from tensorflow.keras.backend import ndim
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def unet(input_size=(512, 512, 1), dropout_perc=0, learning_rate=1E-4):
    """Implements the two-dimensional version of the U-Net dense neural
    network.

    Parameters
    ----------
    input_size : (M, N, C) array-like, optional (default : (512, 512, 1))
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

    # analysis path.
    conv_down_1, max_pool_1 = _analysis_path(inputs,
                                             filters=64,
                                             dropout_perc=dropout_perc)
    conv_down_2, max_pool_2 = _analysis_path(max_pool_1,
                                             filters=128,
                                             dropout_perc=dropout_perc)
    conv_down_3, max_pool_3 = _analysis_path(max_pool_2,
                                             filters=256,
                                             dropout_perc=dropout_perc)
    conv_down_4, max_pool_4 = _analysis_path(max_pool_3,
                                             filters=512,
                                             dropout_perc=0.5)

    # bottleneck.
    conv_down_5 = _analysis_path(max_pool_4,
                                 filters=1024,
                                 dropout_perc=0.5,
                                 is_bottleneck=True)

    # synthesis path.
    conv_up_4 = _synthesis_path(conv_down_5,
                                layer_analysis=conv_down_4,
                                filters=512,
                                dropout_perc=dropout_perc)
    conv_up_3 = _synthesis_path(conv_up_4,
                                layer_analysis=conv_down_3,
                                filters=256,
                                dropout_perc=dropout_perc)
    conv_up_2 = _synthesis_path(conv_up_3,
                                layer_analysis=conv_down_2,
                                filters=128,
                                dropout_perc=dropout_perc)
    conv_up_1 = _synthesis_path(conv_up_2,
                                layer_analysis=conv_down_1,
                                filters=64,
                                dropout_perc=dropout_perc)

    # last layer.
    output, loss = _last_layer_activation(conv_up_1, n_classes=n_classes)
    model = Model(inputs, output)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=['accuracy'])

    return model


def unet_3d(input_size=(64, 64, 64, 1), dropout_perc=0, learning_rate=1E-4):
    """Implements the three-dimensional version of the U-Net dense neural
    network.

    Parameters
    ----------
    input_size : (P, M, N, C) array-like, optional (default : (64, 64, 64, 1))
        Shape of the input data in planes, rows, columns, and channels.
    dropout_perc : float, optional (default : 0)
        Percentage of dropout on each layer.

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

    The analysis path consists of two (3, 3, 3) convolutions each, followed by
    a ReLU. Then, a (2, 2, 2) max pooling with strides of size two. The
    synthesis path, in its turn, consists of an upconvolution with an (2, 2, 2)
    filter, and then two (3, 3, 3) convolutions followed by a ReLU.

    input_size does not contain the batch size. The default input,
    for example, indicates that the model expects images with 64 planes,
    64 rows, 64 columns, and one channel.

    This model is compiled with the optimizer Adam.

    References
    ----------
    .. [1] Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O.
           Ronneberger, “3D U-Net: Learning Dense Volumetric Segmentation
           from Sparse Annotation,” arXiv:1606.06650 [cs], Jun. 2016.

    Examples
    --------
    >>> from models.unet import unet_3d
    >>> model_3d_unet = unet_3d(input_size=(64, 64, 64, 1))
    """
    n_classes = input_size[-1]
    inputs = layers.Input(input_size)

    # analysis path.
    conv_down_1, max_pool_1 = _analysis_path(inputs,
                                             filters=32,
                                             dropout_perc=dropout_perc,
                                             is_unet_3d=True)
    conv_down_2, max_pool_2 = _analysis_path(max_pool_1,
                                             filters=64,
                                             dropout_perc=dropout_perc,
                                             is_unet_3d=True)
    conv_down_3, max_pool_3 = _analysis_path(max_pool_2,
                                             filters=128,
                                             dropout_perc=dropout_perc,
                                             is_unet_3d=True)

    # bottleneck.
    conv_down_4 = _analysis_path(max_pool_3,
                                 filters=256,
                                 dropout_perc=dropout_perc,
                                 is_bottleneck=True,
                                 is_unet_3d=True)


    # synthesis path.
    conv_up_3 = _synthesis_path(conv_down_4,
                                layer_analysis=conv_down_3,
                                filters=512,
                                dropout_perc=dropout_perc,
                                is_unet_3d=True)
    conv_up_2 = _synthesis_path(conv_up_3,
                                layer_analysis=conv_down_2,
                                filters=256,
                                dropout_perc=dropout_perc,
                                is_unet_3d=True)
    conv_up_1 = _synthesis_path(conv_up_2,
                                layer_analysis=conv_down_1,
                                filters=128,
                                dropout_perc=dropout_perc,
                                is_unet_3d=True)

    # last layer.
    output, loss = _last_layer_activation(conv_up_1, n_classes=n_classes)
    model = Model(inputs, output)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=['accuracy'])

    return model


def _analysis_path(layer, filters, dropout_perc=0, is_bottleneck=False,
                   is_unet_3d=False):
    """Apply a conv-BN-ReLU layer with filter size 3, and a max pooling."""
    layer = _conv_bn_relu(layer=layer,
                          filters=filters,
                          kernel_size=3,
                          dropout_perc=dropout_perc)
    if is_unet_3d:
        filters *= 2
    layer = _conv_bn_relu(layer=layer,
                          filters=filters,
                          kernel_size=3,
                          dropout_perc=dropout_perc)

    if is_bottleneck:
        return layer
    else:
        if ndim(layer) == 4:
            max_pool = layers.MaxPooling2D((2, 2))(layer)
        elif ndim(layer) == 5:
            max_pool = layers.MaxPooling3D((2, 2, 2))(layer)
        return layer, max_pool


def _conv_bn_relu(layer, filters, kernel_size=3, dropout_perc=0):
    """Apply successively Convolution, Batch Normalization, ReLU nonlinearity
    and Dropout, when dropout_perc > dropout_perc."""
    if ndim(layer) == 4:
        layer = layers.Conv2D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              kernel_initializer='he_uniform')(layer)
    elif ndim(layer) == 5:
        layer = layers.Conv3D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              kernel_initializer='he_uniform')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.Activation('relu')(layer)

    if dropout_perc != 0:
        layer = layers.Dropout(dropout_perc)(layer)

    return layer


def _last_layer_activation(layer, n_classes=1):
    """Performs a convolution with kernel size 1, followed by the an activation
    chosen according to the input size.

    Notes
    -----
    For a 2D input, it includes a 2D convolution before this conv, representing
    the output segmentation map described in Ronneberger et al (2015).
    """
    if ndim(layer) == 4:
        layer = layers.Conv2D(filters=2,
                              kernel_size=3,
                              padding='same',
                              kernel_initializer='he_uniform')(layer)
        layer = layers.BatchNormalization()(layer)
        layer = layers.Activation('relu')(layer)

        layer = layers.Conv2D(filters=n_classes,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer='he_uniform')(layer)
    elif ndim(layer) == 5:
        layer = layers.Conv3D(filters=n_classes,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer='he_uniform')(layer)

    if n_classes == 1:
        output = layers.Activation('sigmoid')(layer)
        loss = 'binary_crossentropy'
    else:
        output = layers.Activation('softmax')(layer)
        loss = 'categorical_crossentropy'

    return output, loss


def _synthesis_path(layer, layer_analysis, filters, dropout_perc=0,
                    is_unet_3d=False):
    """
    """
    layer = _upconv_bn_relu(layer, filters, kernel_size=2)
    merge = layers.concatenate([layer_analysis, layer], axis=-1)

    if is_unet_3d:
        filters = int(filters/2)
    layer = _conv_bn_relu(layer=merge,
                          filters=filters,
                          kernel_size=3,
                          dropout_perc=dropout_perc)
    layer = _conv_bn_relu(layer=layer,
                          filters=filters,
                          kernel_size=3,
                          dropout_perc=dropout_perc)

    return layer


def _upconv_bn_relu(layer, filters, kernel_size=2):
    """
    """
    if ndim(layer) == 4:
        layer = layers.Conv2D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              kernel_initializer='he_uniform')(
                                  layers.UpSampling2D(size=(2, 2))
                                  (layer)
                             )
    elif ndim(layer) == 5:
        layer = layers.Conv3D(filters=filters,
                              kernel_size=kernel_size,
                              padding='same',
                              kernel_initializer='he_uniform')(
                                  layers.UpSampling3D(size=(2, 2, 2))
                                  (layer)
                             )
    layer = layers.BatchNormalization()(layer)
    layer = layers.Activation('relu')(layer)

    return layer

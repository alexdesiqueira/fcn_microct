from tensorflow.keras import layers
from tensorflow.keras.backend import ndim
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def tiramisu(input_size=(256, 256, 1), preset_model='tiramisu-67',
             dropout_perc=0.2):
    """Implements the One Hundred Layers Tiramisu dense neural network.

    Tiramisu is a full convolutional network based on DenseNets, for
    applications on semantic segmentation.

    Parameters
    ----------
    input_size : (M, N, C) array-like, optional (default : (256, 256, 1))
        Shape of the input data in rows, columns, and channels.
    preset_model : string, optional (default : 'tiramisu-67')
        Name of the preset Tiramisu model. Options are 'tiramisu-56',
        'tiramisu-67', and 'tiramisu-103'.
    dropout_perc : float (default : 0.2)
        Percentage to dropout, i.e. neurons to ignore during training.

    Returns
    -------
    model : model class
        A Keras model class with its methods.

    Notes
    -----
    input_size does not contain the batch size. The default input,
    for example, indicates that the model expects images with 256 rows,
    256 columns, and one channel.

    This model is compiled with the optimizer RMSprop, according to _[1].

    References
    ----------
    .. [1] S. Jégou, M. Drozdzal, D. Vazquez, A. Romero, and Y. Bengio,
           “The One Hundred Layers Tiramisu: Fully Convolutional DenseNets
           for Semantic Segmentation,” arXiv:1611.09326 [cs], Oct. 2017.
    .. [2] https://github.com/smdYe/FC-DenseNet-Keras
    .. [3] https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py
    .. [4] https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/

    Examples
    --------
    >>> from models import tiramisu
    >>> model_tiramisu56 = tiramisu(input_size=(128, 128, 1),
                                    preset_model='tiramisu-56')
    """
    n_classes = input_size[-1]
    parameters = _tiramisu_parameters(preset_model)
    filters_first_conv, pool, growth_rate, layers_per_block = parameters.values()

    if type(layers_per_block) == list:
        _check_list_input(layers_per_block, pool)
    elif type(layers_per_block) == int:
        layers_per_block = [layers_per_block] * (2 * pool + 1)
    else:
        raise ValueError

    # first convolution.
    inputs = layers.Input(input_size)
    stack = layers.Conv2D(filters=filters_first_conv,
                          kernel_size=3,
                          padding='same',
                          kernel_initializer='he_uniform')(inputs)
    n_filters = filters_first_conv

    # downsampling path.
    skip_connection = []

    for idx in range(pool):
        for _ in range(layers_per_block[idx]):
            layer = _bn_relu_conv(stack,
                                  n_filters=growth_rate,
                                  dropout_perc=dropout_perc)
            stack = layers.concatenate([stack, layer])
            n_filters += growth_rate

        skip_connection.append(stack)
        stack = _transition_down(stack,
                                 n_filters=n_filters,
                                 dropout_perc=dropout_perc)
    skip_connection = skip_connection[::-1]

    # bottleneck.
    upsample_block = []

    for _ in range(layers_per_block[pool]):
        layer = _bn_relu_conv(stack,
                              n_filters=growth_rate,
                              dropout_perc=dropout_perc)
        upsample_block.append(layer)
        stack = layers.concatenate([stack, layer])
    upsample_block = layers.concatenate(upsample_block)

    # upsampling path.
    for idx in range(pool):
        filters_to_keep = growth_rate * layers_per_block[pool + idx]
        stack = _transition_up(skip_connection[idx],
                               upsample_block,
                               filters_to_keep)

        upsample_block = []
        for _ in range(layers_per_block[pool + idx + 1]):
            layer = _bn_relu_conv(stack,
                                  growth_rate,
                                  dropout_perc=dropout_perc)
            upsample_block.append(layer)
            stack = layers.concatenate([stack, layer])

        upsample_block = layers.concatenate(upsample_block)

    # applying the sigmoid layer.
    output = _last_layer_activation(stack, n_classes=1)
    model = Model(inputs, output)

    if n_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(optimizer=RMSprop(learning_rate=1e-5),
                  loss=loss,
                  metrics=['accuracy'])

    return model


def tiramisu_3d(input_size=(32, 32, 32, 1), preset_model='tiramisu-67',
                dropout_perc=0.2):
    """Implements the three-dimensional version of the One Hundred
    Layers Tiramisu dense neural network.

    Tiramisu is a full convolutional network based on DenseNets, for
    applications on semantic segmentation.

    Parameters
    ----------
    input_size : (P, M, N, C) array-like, optional (default : (32, 32, 32, 1))
        Shape of the input data in planes, rows, columns, and channels.
    preset_model : string, optional (default : 'tiramisu-67')
        Name of the preset Tiramisu model. Options are 'tiramisu-56',
        'tiramisu-67', and 'tiramisu-103'.
    dropout_perc : float (default : 0.2)
        Percentage to dropout, i.e. neurons to ignore during training.

    Returns
    -------
    model : model class
        A Keras model class with its methods.

    Notes
    -----
    input_size does not contain the batch size. The default input,
    for example, indicates that the model expects images with 32 planes,
    32 rows, 32 columns, and one channel.

    This model is compiled with the optimizer RMSprop, according to _[1],
    and written only to one channel.

    References
    ----------
    .. [1] S. Jégou, M. Drozdzal, D. Vazquez, A. Romero, and Y. Bengio,
           “The One Hundred Layers Tiramisu: Fully Convolutional DenseNets
           for Semantic Segmentation,” arXiv:1611.09326 [cs], Oct. 2017.
    .. [2] https://github.com/smdYe/FC-DenseNet-Keras
    .. [3] https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py
    .. [4] https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/

    Examples
    --------
    >>> from models import tiramisu
    >>> model_tiramisu56_3d = tiramisu(input_size=(32, 32, 32, 1),
                                       preset_model='tiramisu-56')
    """
    n_classes = input_size[-1]
    parameters = _tiramisu_parameters(preset_model)
    filters_first_conv, pool, growth_rate, layers_per_block = parameters.values()

    if type(layers_per_block) == list:
        _check_list_input(layers_per_block, pool)
    elif type(layers_per_block) == int:
        layers_per_block = [layers_per_block] * (2 * pool + 1)
    else:
        raise ValueError

    # first convolution.
    inputs = layers.Input(input_size)
    stack = layers.Conv3D(filters=filters_first_conv,
                          kernel_size=3,
                          padding='same',
                          kernel_initializer='he_uniform')(inputs)
    n_filters = filters_first_conv

    # downsampling path.
    skip_connection = []

    for idx in range(pool):
        for _ in range(layers_per_block[idx]):
            layer = _bn_relu_conv(stack,
                                  n_filters=growth_rate,
                                  dropout_perc=dropout_perc)
            stack = layers.concatenate([stack, layer])
            n_filters += growth_rate

        skip_connection.append(stack)
        stack = _transition_down(stack,
                                 n_filters=n_filters,
                                 dropout_perc=dropout_perc)
    skip_connection = skip_connection[::-1]

    # bottleneck.
    upsample_block = []

    for _ in range(layers_per_block[pool]):
        layer = _bn_relu_conv(stack,
                              n_filters=growth_rate,
                              dropout_perc=dropout_perc)
        upsample_block.append(layer)
        stack = layers.concatenate([stack, layer])
    upsample_block = layers.concatenate(upsample_block)

    # upsampling path.
    for idx in range(pool):
        filters_to_keep = growth_rate * layers_per_block[pool + idx]
        stack = _transition_up_3d(skip_connection[idx],
                                  upsample_block,
                                  filters_to_keep)

        upsample_block = []
        for _ in range(layers_per_block[pool + idx + 1]):
            layer = _bn_relu_conv(stack,
                                  growth_rate,
                                  dropout_perc=dropout_perc)
            upsample_block.append(layer)
            stack = layers.concatenate([stack, layer])

        upsample_block = layers.concatenate(upsample_block)

    # applying the last layer.
    output = _last_layer_activation(stack, n_classes=1)
    model = Model(inputs, output)

    if n_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(optimizer=RMSprop(learning_rate=1e-5),
                  loss=loss,
                  metrics=['accuracy'])

    return model


def _bn_relu_conv(inputs, n_filters, filter_size=3, dropout_perc=0.2):
    """Apply successively Batch Normalization, ReLU nonlinearity,
    Convolution and Dropout, when dropout_perc > 0."""
    layer = layers.BatchNormalization()(inputs)
    layer = layers.Activation('relu')(layer)
    if ndim(inputs) == 4:
        layer = layers.Conv2D(n_filters,
                              filter_size,
                              padding='same',
                              kernel_initializer='he_uniform')(layer)
    elif ndim(inputs) == 5:
        layer = layers.Conv3D(n_filters,
                              filter_size,
                              padding='same',
                              kernel_initializer='he_uniform')(layer)
    if dropout_perc != 0:
        layer = layers.Dropout(dropout_perc)(layer)
    return layer


def _check_list_input(layers_per_block, pool):
    """Check if the layers list has the correct size."""
    assert len(layers_per_block) == 2 * pool + 1


def _last_layer_activation(inputs, n_classes=1):
    """Performs 1x1 convolution followed by activation."""
    if ndim(inputs) == 4:
        layer = layers.Conv2D(n_classes,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer='he_uniform')(inputs)
    elif ndim(inputs) == 5:
        layer = layers.Conv3D(n_classes,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer='he_uniform')(inputs)

    if n_classes == 1:
        output = layers.Activation('sigmoid')(layer)

    else:
        output = layers.Activation('softmax')(layer)

    return output


def _tiramisu_parameters(preset_model='tiramisu-67'):
    """Returns Tiramisu parameters based on the chosen model."""
    if preset_model == 'tiramisu-56':
        parameters = {
            'filters_first_conv': 48,
            'pool': 5,
            'growth_rate': 12,
            'layers_per_block': 4
        }
    elif preset_model == 'tiramisu-67':
        parameters = {
            'filters_first_conv': 48,
            'pool': 5,
            'growth_rate': 16,
            'layers_per_block': 5
        }
    elif preset_model == 'tiramisu-103':
        parameters = {
            'filters_first_conv': 48,
            'pool': 5,
            'growth_rate': 16,
            'layers_per_block': [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
        }
    else:
        raise ValueError(f'Tiramisu model {preset_model} not available.')
    return parameters


def _transition_down(inputs, n_filters, dropout_perc=0.2):
    """ Apply a BN-ReLU-conv layer with filter size 1, and a max pooling."""
    layer = _bn_relu_conv(inputs,
                          n_filters,
                          filter_size=1,
                          dropout_perc=dropout_perc)
    if ndim(inputs) == 4:
        layer = layers.MaxPooling2D((2, 2))(layer)
    elif ndim(inputs) == 5:
        layer = layers.MaxPooling3D((2, 2, 2))(layer)
    return layer


def _transition_up(skip_connection, block_to_upsample, filters_to_keep):
    '''Performs upsampling on block_to_upsample by a factor 2 and
    concatenates it with the skip_connection'''
    # Upsample and concatenate with skip connection
    layer = layers.Conv2DTranspose(filters_to_keep,
                                   kernel_size=3,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer='he_uniform')(
                                       block_to_upsample
                                   )
    layer = layers.concatenate([layer, skip_connection], axis=-1)
    return layer


def _transition_up_3d(skip_connection, block_to_upsample, filters_to_keep):
    '''Performs upsampling on block_to_upsample by a factor 2 and
    concatenates it with the skip_connection'''
    # Upsample and concatenate with skip connection
    layer = layers.Conv3DTranspose(filters_to_keep,
                                   kernel_size=3,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer='he_uniform')(
                                       block_to_upsample
                                   )
    layer = layers.concatenate([layer, skip_connection], axis=-1)
    return layer

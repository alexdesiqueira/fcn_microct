from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def tiramisu_3d(input_size=(32, 32, 32, 1), preset_model='tiramisu_67',
                dropout_perc=0.2):
    """ Implements the three-dimensional version of the One Hundred
    Layers Tiramisu dense neural network.
    """
    if preset_model == 'tiramisu_56':
        filters_first_conv = 48
        pool = 5
        growth_rate = 12
        layers_per_block = 4
    elif preset_model == 'tiramisu_67':
        filters_first_conv = 48
        pool = 5
        growth_rate = 16
        layers_per_block = 5
    elif preset_model == 'tiramisu_103':
        filters_first_conv = 48
        pool = 5
        growth_rate = 16
        layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
        raise ValueError(f'Tiramisu model {preset_model} not available.')

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
            layer = bn_relu_conv_3d(stack,
                                    n_filters=growth_rate,
                                    dropout_perc=dropout_perc)
            stack = layers.concatenate([stack, layer])
            n_filters += growth_rate

        skip_connection.append(stack)
        stack = transition_down_3d(stack,
                                   n_filters=n_filters,
                                   dropout_perc=dropout_perc)
    skip_connection = skip_connection[::-1]

    # bottleneck.
    upsample_block = []

    for _ in range(layers_per_block[pool]):
        layer = bn_relu_conv_3d(stack,
                                n_filters=growth_rate,
                                dropout_perc=dropout_perc)
        upsample_block.append(layer)
        stack = layers.concatenate(upsample_block)
    upsample_block = layers.concatenate(upsample_block)

    # upsampling path.
    for idx in range(pool):
        filters_to_keep = growth_rate * layers_per_block[pool + idx]
        stack = transition_up_3d(skip_connection[idx],
                                 upsample_block,
                                 filters_to_keep)

        upsample_block = []
        for _ in range(layers_per_block[pool + idx + 1]):
            layer = bn_relu_conv_3d(stack,
                                    growth_rate,
                                    dropout_perc=dropout_perc)
            upsample_block.append(layer)
            stack = layers.concatenate([stack, layer])

        upsample_block = layers.concatenate(upsample_block)

    # applying the sigmoid layer.
    output = sigmoid_layer_3d(stack, n_classes=1)
    model = Model(inputs, output)

    model.compile(optimizer=RMSprop(learning_rate=1e-5),  # was : 1e-4
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def bn_relu_conv_3d(inputs, n_filters, filter_size=3, dropout_perc=0.2):
    '''Apply successively Batch Normalization, ReLu nonlinearity,
    Convolution and Dropout (if dropout_perc > 0)'''
    layer = layers.BatchNormalization()(inputs)
    layer = layers.Activation('relu')(layer)
    layer = layers.Conv3D(n_filters,
                          filter_size,
                          padding='same',
                          kernel_initializer='he_uniform')(layer)
    if dropout_perc != 0:
        layer = layers.Dropout(dropout_perc)(layer)
    return layer


def sigmoid_layer_3d(inputs, n_classes=1):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """
    layer = layers.Conv3D(n_classes,
                          kernel_size=1,
                          padding='same',
                          kernel_initializer='he_uniform')(inputs)
    output = layers.Activation('sigmoid')(layer)  # or softmax for multi-class
    return output


def transition_down_3d(inputs, n_filters, dropout_perc=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and
    a max pooling with a factor 2  """
    layer = bn_relu_conv_3d(inputs,
                            n_filters,
                            filter_size=1,
                            dropout_perc=dropout_perc)
    layer = layers.MaxPooling3D((2, 2, 2))(layer)
    return layer


def transition_up_3d(skip_connection, block_to_upsample, filters_to_keep):
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


def _check_list_input(layers_per_block, pool):
    """Check if the layers list has the correct size."""
    assert len(layers_per_block) == 2 * pool + 1

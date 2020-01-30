from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def tiramisu(input_size=(512, 512, 1), preset_model='FC-DenseNet103',
             dropout_perc=0.2):
    """Implements the One Hundred Layers Tiramisu dense neural network.

    Notes
    -----
    Adapted from: <https://github.com/smdYe/FC-DenseNet-Keras>.
    <https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py>

    References
    ----------
    [1] <https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/>
    """
    if preset_model == 'FC-DenseNet56':
        filters_first_conv = 48
        pool = 5
        growth_rate = 12
        layers_per_block = 4
    elif preset_model == 'FC-DenseNet67':
        filters_first_conv = 48
        pool = 5
        growth_rate = 16
        layers_per_block = 5
    elif preset_model == 'FC-DenseNet103':
        filters_first_conv = 48
        pool = 5
        growth_rate = 16
        layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
        raise ValueError(f'Unsupported FC-DenseNet model {preset_model}.')

    if type(layers_per_block) == list:
        _check_list_input(layers_per_block, pool)
    elif type(layers_per_block) == int:
        layers_per_block = [layers_per_block] * (2 * pool + 1)
    else:
        raise ValueError

    # First convolution.
    inputs = layers.Input(input_size)
    stack = layers.Conv2D(filters=filters_first_conv,
                          kernel_size=3,
                          padding='same',
                          kernel_initializer='he_uniform')(inputs)
    n_filters = filters_first_conv

    # Downsampling path.
    skip_connection = []

    for i in range(pool):
        for j in range(layers_per_block[i]):
            layer = layer_bn_relu_conv(stack,
                                       n_filters=growth_rate,
                                       dropout_perc=dropout_perc)
            stack = layers.concatenate([stack, layer])
            n_filters += growth_rate

        skip_connection.append(stack)
        stack = transition_down(stack,
                                n_filters=n_filters,
                                dropout_perc=dropout_perc)
    skip_connection = skip_connection[::-1]

    # Bottleneck.
    upsample_block = []

    for j in range(layers_per_block[pool]):
        layer = layer_bn_relu_conv(stack,
                                   n_filters=growth_rate,
                                   dropout_perc=dropout_perc)
        upsample_block.append(layer)
        stack = layers.concatenate([stack, layer])
    upsample_block = layers.concatenate(upsample_block)

    # Upsampling path.
    for i in range(pool):
        filters_to_keep = growth_rate * layers_per_block[pool + i]
        stack = transition_up(skip_connection[i],
                              upsample_block,
                              filters_to_keep)

        upsample_block = []
        for j in range(layers_per_block[pool + i + 1]):
            layer = layer_bn_relu_conv(stack,
                                       growth_rate,
                                       dropout_perc=dropout_perc)
            upsample_block.append(layer)
            stack = layers.concatenate([stack, layer])

        upsample_block = layers.concatenate(upsample_block)

    # Applying the sigmoid layer.
    output = sigmoid_layer(stack, n_classes=1)
    model = Model(inputs, output)

    model.compile(optimizer=RMSprop(learning_rate=1e-5),  # was : 1e-4
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def layer_bn_relu_conv(inputs, n_filters, filter_size=3, dropout_perc=0.2):
    '''Apply successively Batch Normalization, ReLu nonlinearity,
    Convolution and Dropout (if dropout_perc > 0)'''
    layer = layers.BatchNormalization()(inputs)
    layer = layers.Activation('relu')(layer)
    layer = layers.Conv2D(n_filters,
                          filter_size,
                          padding='same',
                          kernel_initializer='he_uniform')(layer)
    if dropout_perc != 0:
        layer = layers.Dropout(dropout_perc)(layer)
    return layer


def sigmoid_layer(inputs, n_classes=1):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """
    layer = layers.Conv2D(n_classes,
                          kernel_size=1,
                          padding='same',
                          kernel_initializer='he_uniform')(inputs)
    output = layers.Activation('sigmoid')(layer)  # or softmax for multi-class
    return output


def transition_down(inputs, n_filters, dropout_perc=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and
    a max pooling with a factor 2  """
    layer = layer_bn_relu_conv(inputs,
                               n_filters,
                               filter_size=1,
                               dropout_perc=dropout_perc)
    layer = layers.MaxPooling2D((2, 2))(layer)
    return layer


def transition_up(skip_connection, block_to_upsample, filters_to_keep):
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


def _check_list_input(layers_per_block, pool):
    """Check if the layers list has the correct size."""
    assert len(layers_per_block) == 2 * pool + 1

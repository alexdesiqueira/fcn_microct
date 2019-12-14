from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def tiramisu(input_size=(512, 512, 1), pool=5, growth_rate=16, dropout_p=0.2, layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]):
    """
    Adapted from https://github.com/smdYe/FC-DenseNet-Keras/.
    """
    n_filters = 48  # terrible. Fix it later
    inputs = layers.Input(input_size)
    stack = layers.Conv2D(filters=48,
                          kernel_size=3,
                          padding='same',
                          kernel_initializer='he_uniform')(inputs)

    # Downsampling path.
    skip_connection = []
    for i in range(pool):
        for j in range(layers_per_block[i]):
            layer = BN_ReLU_Conv(stack,
                                 n_filters=growth_rate,
                                 dropout_p=dropout_p)
            stack = layers.concatenate([stack, layer])
            n_filters += growth_rate

        skip_connection.append(stack)
        stack = TransitionDown(stack, n_filters, dropout_p=dropout_p)
    skip_connection = skip_connection[::-1]

    # Bottleneck.
    upsample_block = []

    for j in range(layers_per_block[pool]):
        layer = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        upsample_block.append(layer)
        stack = layers.concatenate([stack, layer])
    upsample_block = layers.concatenate(upsample_block)

    # Upsampling path.
    for i in range(pool):
        filters_to_keep = growth_rate * layers_per_block[pool + i]
        stack = TransitionUp(skip_connection[i], upsample_block, filters_to_keep)

        upsample_block = []
        for j in range(layers_per_block[pool + i + 1]):
            layer = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            upsample_block.append(layer)
            stack = layers.concatenate([stack, layer])

        upsample_block = layers.concatenate(upsample_block)

    # Softmax
    output = SoftmaxLayer(stack, n_classes=1)  # is 1 correct?
    model = Model(inputs, output)

    model.compile(optimizer=Adam(learning_rate=1e-5),  # was : 1e-4
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)'''

    l = layers.BatchNormalization()(inputs)
    l = layers.Activation('relu')(l)
    l = layers.Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
    if dropout_p != 0.0:
        l = layers.Dropout(dropout_p)(l)
    return l


def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = layers.MaxPooling2D((2, 2))(l)
    return l


def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
    #Upsample and concatenate with skip connection
    l = layers.Conv2DTranspose(n_filters_keep,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               kernel_initializer='he_uniform')(block_to_upsample)
    l = layers.concatenate([l, skip_connection], axis=-1)
    return l


def SoftmaxLayer(inputs, n_classes):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """
    l = layers.Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform')(inputs)
#    l = Reshape((-1, n_classes))(l)
    l = layers.Activation('sigmoid')(l)#or softmax for multi-class
    return l

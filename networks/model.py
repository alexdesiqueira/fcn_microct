from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    """Implements the U-Net dense neural network.
    """
    inputs = layers.Input(input_size)
    conv1 = layers.Conv2D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2,
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal'
                        )(layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv6)

    up7 = layers.Conv2D(256, 2,
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv7)

    up8 = layers.Conv2D(128, 2,
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal'
                        )(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv8)

    up9 = layers.Conv2D(64, 2,
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal'
                        )(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(conv9)
    conv10 = layers.Conv2D(1, 1,
                           activation='sigmoid')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(learning_rate=1e-5),  # was : 1e-4
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def unet_3d(input_size=(64, 64, 64, 1)):
    """Implements the 3D U-Net dense neural network.

    Notes
    -----
        The implementation follows Çiçek et al. (2016):

    N -> 32 -> 64  ===============================================================================>>  64+128 -> 64 -> 64 ~> N
                \                                                                                     /
                64 -> 64 -> 128  =============================================>>  128+256 -> 128 -> 128
                              \                                                   /
                              128 -> 128 -> 256  =========>>  256+512 -> 256 -> 256
                                              \               /
                                              256 -> 256 -> 512

    where:
    =>> : concat
    ~> : conv
    -> : conv + BN + ReLu
    \ : max pool
    / : up-conv
    """
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
    merge_3 = layers.concatenate([conv_down_3, conv_up_3], axis=-1)  # was: axis=4
    conv_up_3 = layers.Conv3D(filters=256,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(merge_3)

    # level 2 - up
    conv_up_2 = layers.Conv3D(filters=256,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling3D(size=(2, 2, 2))
                                  (conv_up_3)
                              )
    merge_2 = layers.concatenate([conv_down_2, conv_up_2], axis=-1)  # was: axis=4
    conv_up_2 = layers.Conv3D(filters=128,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(merge_2)

    # level 1 - up
    conv_up_1 = layers.Conv3D(filters=128,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling3D(size=(2, 2, 2))
                                  (conv_up_2)
                              )
    merge_1 = layers.concatenate([conv_down_1, conv_up_1], axis=-1)  # was: axis=4
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
    conv_output = layers.Conv3D(filters=1,
                                kernel_size=1,
                                activation='sigmoid')(conv_up_1)

    model = Model(inputs, conv_output)

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',  # maybe we need to change it
                  metrics=['accuracy'])

    return model


def tiramisu(input_size=(512, 512, 1), filters_first_conv=48, pool=5,
             growth_rate=16, dropout_perc=0.2,
             layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]):
    """Implements the One Hundred Layers Tiramisu dense neural network.

    Notes
    -----
    Adapted from https://github.com/smdYe/FC-DenseNet-Keras
    /.
    """
    if type(layers_per_block) == list:
        print(len(layers_per_block))
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
            layer = BN_ReLU_Conv(stack,
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
        layer = BN_ReLU_Conv(stack,
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
            layer = BN_ReLU_Conv(stack,
                                 growth_rate,
                                 dropout_perc=dropout_perc)
            upsample_block.append(layer)
            stack = layers.concatenate([stack, layer])

        upsample_block = layers.concatenate(upsample_block)

    # Applying the softmax layer.
    output = softmax_layer(stack, n_classes=1)  # is 1 correct?
    model = Model(inputs, output)

    model.compile(optimizer=Adam(learning_rate=1e-5),  # was : 1e-4
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_perc=0.2):
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


def transition_down(inputs, n_filters, dropout_perc=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and
    a max pooling with a factor 2  """
    layer = BN_ReLU_Conv(inputs,
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


def softmax_layer(inputs, n_classes):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size * n_rows * n_cols, n_classes)
    """
    layer = layers.Conv2D(n_classes,
                          kernel_size=1,
                          padding='same',
                          kernel_initializer='he_uniform')(inputs)
    layer = layers.Activation('sigmoid')(layer)  # or softmax for multi-class
    return layer

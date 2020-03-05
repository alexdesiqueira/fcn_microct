from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def unet(input_size=(256, 256, 1)):
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
                        kernel_initializer='he_normal')(
                            layers.UpSampling2D(size=(2, 2))(conv6)
                        )
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
    merge_2 = layers.concatenate([conv_down_2, conv_up_2], axis=-1)
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
    conv_output = layers.Conv3D(filters=1,
                                kernel_size=1,
                                activation='sigmoid')(conv_up_1)

    model = Model(inputs, conv_output)

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',  # maybe we need to change it
                  metrics=['accuracy'])

    return model


def unet_3d_3(input_size=(64, 64, 64, 1)):
    """Implements the 3D U-Net dense neural network.

    Notes
    -----
        The implementation follows Çiçek et al. (2016):

    N -> 32 -> 64  ===============================================================================>>  64+128 -> 64 -> 64 ~> N
                \                                                                                     /
                64 -> 64 -> 128  =============================================>>  128+256 -> 128 -> 128
                              \                                                   /
                                              \               /
                                              128 -> 128 -> 256

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

    # level 2 - up
    conv_up_2 = layers.Conv3D(filters=256,
                              kernel_size=3,
                              activation='relu',
                              padding='same',
                              kernel_initializer='he_normal')(
                                  layers.UpSampling3D(size=(2, 2, 2))
                                  (conv_down_3)
                              )
    merge_2 = layers.concatenate([conv_down_2, conv_up_2], axis=-1)
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
    conv_output = layers.Conv3D(filters=1,
                                kernel_size=1,
                                activation='sigmoid')(conv_up_1)

    model = Model(inputs, conv_output)

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',  # maybe we need to change it
                  metrics=['accuracy'])

    return model
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


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

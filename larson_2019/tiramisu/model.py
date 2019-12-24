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


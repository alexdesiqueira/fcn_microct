from tensorflow.keras.applications import densenet as ds
from tensorflow.keras.optimizers import Adam, SGD


def densenet(input_size=(512, 512, 1), densenet_model='201'):
    """
    https://keras.io/applications/#densenet
    """

    args = dict(weights=None,
                input_shape=input_size,
		classes=2)

    models = {
        '121': ds.DenseNet121(**args),
        '169': ds.DenseNet169(**args),
        '201': ds.DenseNet201(**args)
    }
    model = models.get(densenet_model, None)

    if model is None:
        raise

    model.compile(optimizer=SGD(learning_rate=1e-2),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

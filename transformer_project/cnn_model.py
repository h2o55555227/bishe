from tensorflow import keras
from tensorflow.keras import layers


def cnn_block(x, filters, kernel_size, activation, dropout_rate):
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", activation=activation)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_cnn_model(
    input_shape,
    activation="relu",
    filters=[64, 128, 256],
    kernel_size=3,
    dropout_rate=0.1,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    for filter_size in filters:
        x = cnn_block(
            x,
            filters=filter_size,
            kernel_size=kernel_size,
            activation=activation,
            dropout_rate=dropout_rate,
        )
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)

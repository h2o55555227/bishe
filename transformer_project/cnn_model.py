from tensorflow import keras
from tensorflow.keras import layers


def cnn_block(x, filters, kernel_size, activation, dropout_rate):
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_cnn_model(
    input_shape,
    activation="relu",
    filters=[32, 64],
    kernel_size=5,
    dropout_rate=0.3,
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
    
    # 添加额外的卷积层以捕获更多时序特征
    x = layers.Conv1D(filters=filters[-1], kernel_size=kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation=activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)

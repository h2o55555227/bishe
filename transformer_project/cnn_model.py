from tensorflow import keras
from tensorflow.keras import layers


def cnn_block(x, filters, kernel_size, activation, dropout_rate, use_batch_norm, residual_connection):
    residual = x
    
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
    
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation(activation)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 添加残差连接
    if residual_connection and residual.shape[-1] == filters:
        residual = layers.MaxPooling1D(pool_size=2)(residual)
        x = layers.Add()([x, residual])
    
    return x


def build_cnn_model(
    input_shape,
    activation="relu",
    filters=[32, 64],
    kernel_size=3,
    dropout_rate=0.15,
    use_batch_norm=False,
    residual_connection=False,
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
            use_batch_norm=use_batch_norm,
            residual_connection=residual_connection,
        )
    
    # 添加额外的卷积层以捕获更多时序特征
    x = layers.Conv1D(filters=filters[-1], kernel_size=kernel_size, padding="same")(x)
    
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation=activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)

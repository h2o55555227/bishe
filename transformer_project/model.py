from tensorflow import keras
from tensorflow.keras import layers


def transformer_block(
    x, num_heads, key_dim, ff_dim, projection_dim, activation, dropout_rate
):
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
    )(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    x = layers.LayerNormalization()(x + attn_output)

    ff = layers.Dense(ff_dim, activation=activation)(x)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(projection_dim)(ff)
    ff = layers.Dropout(dropout_rate)(ff)
    return layers.LayerNormalization()(x + ff)


def build_transformer_model(
    input_shape,
    activation="gelu",
    projection_dim=128,
    num_heads=8,
    ff_dim=256,
    num_transformer_blocks=2,
    dropout_rate=0.1,
):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(projection_dim)(inputs)
    x = layers.Dropout(dropout_rate)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_block(
            x,
            num_heads=num_heads,
            key_dim=projection_dim,
            ff_dim=ff_dim,
            projection_dim=projection_dim,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs)

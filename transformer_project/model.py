from tensorflow import keras
from tensorflow.keras import layers


def transformer_block(x, num_heads, key_dim, ff_dim, projection_dim, activation):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = layers.LayerNormalization()(x + attn_output)

    ff = layers.Dense(ff_dim, activation=activation)(x)
    ff = layers.Dense(projection_dim)(ff)
    return layers.LayerNormalization()(x + ff)


def build_transformer_model(
    input_shape,
    activation="gelu",
    projection_dim=128,
    num_heads=8,
    ff_dim=256,
    num_transformer_blocks=2,
):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(projection_dim)(inputs)

    for _ in range(num_transformer_blocks):
        x = transformer_block(
            x,
            num_heads=num_heads,
            key_dim=projection_dim,
            ff_dim=ff_dim,
            projection_dim=projection_dim,
            activation=activation,
        )

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs)

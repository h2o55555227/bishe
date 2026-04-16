from tensorflow import keras
from tensorflow.keras import layers


def build_transformer_model(input_shape, activation="relu"):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(64)(inputs)

    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.LayerNormalization()(x + attn_output)

    ff = layers.Dense(128, activation=activation)(x)
    ff = layers.Dense(64)(ff)
    x = layers.LayerNormalization()(x + ff)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs)

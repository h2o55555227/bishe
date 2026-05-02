import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def transformer_block(
    x,
    num_heads,
    projection_dim,
    ff_dim,
    activation="swish",
    dropout_rate=0.05,
):
    head_dim = projection_dim // num_heads

    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_dim,
        dropout=dropout_rate,
    )(x, x)

    attn = layers.Dropout(dropout_rate)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ff = layers.Dense(ff_dim, activation=activation)(x)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(projection_dim)(ff)
    ff = layers.Dropout(dropout_rate)(ff)

    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def build_transformer_model(
    input_shape,
    activation="swish",
    projection_dim=160,
    num_heads=8,
    ff_dim=640,
    num_transformer_blocks=4,
    dropout_rate=0.03,
    patch_size=3,
):
    inputs = keras.Input(shape=input_shape)

    conv3 = layers.Conv1D(
        96,
        kernel_size=3,
        padding="same",
        activation=activation,
    )(inputs)
    conv5 = layers.Conv1D(
        96,
        kernel_size=5,
        padding="same",
        activation=activation,
    )(inputs)
    conv_dilated = layers.Conv1D(
        96,
        kernel_size=3,
        dilation_rate=3,
        padding="same",
        activation=activation,
    )(inputs)

    x = layers.Concatenate()([conv3, conv5, conv_dilated])
    x = layers.Conv1D(
        projection_dim,
        kernel_size=1,
        padding="same",
        activation=activation,
    )(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(dropout_rate)(x)

    seq_len = input_shape[0]
    num_patches = seq_len // patch_size
    trimmed_seq_len = num_patches * patch_size
    if trimmed_seq_len != seq_len:
        x = layers.Cropping1D(cropping=(0, seq_len - trimmed_seq_len))(x)

    x = layers.Reshape((num_patches, patch_size * projection_dim))(x)
    x = layers.Dense(projection_dim)(x)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim,
    )(positions)
    x = x + pos_embedding[tf.newaxis, ...]
    x = layers.Dropout(dropout_rate)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_block(
            x,
            num_heads=num_heads,
            projection_dim=projection_dim,
            ff_dim=ff_dim,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(96, activation=activation)(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

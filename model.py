import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ===============================
# Transformer Encoder Block
# ===============================
def transformer_block(
    x,
    num_heads,
    projection_dim,
    ff_dim,
    activation="swish",
    dropout_rate=0.05,
):
    # 更合理的 head_dim
    head_dim = projection_dim // num_heads

    # Multi-Head Self Attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_dim,
        dropout=dropout_rate,
    )(x, x)

    attn = layers.Dropout(dropout_rate)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed Forward Network
    ff = layers.Dense(ff_dim, activation=activation)(x)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(projection_dim)(ff)
    ff = layers.Dropout(dropout_rate)(ff)

    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    return x


# ===============================
# Build Final Improved Model
# ===============================
def build_transformer_model(
    input_shape,
    activation="swish",
    projection_dim=128,
    num_heads=4,
    ff_dim=512,
    num_transformer_blocks=3,
    dropout_rate=0.05,
    patch_size=6,
):
    """
    input_shape = (120, 18)
    """

    inputs = keras.Input(shape=input_shape)

    # ==================================
    # 1. CNN Local Feature Extractor
    # ==================================
    x = layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation=activation
    )(inputs)

    x = layers.Conv1D(
        128,
        kernel_size=5,
        padding="same",
        activation=activation
    )(x)

    x = layers.Conv1D(
        projection_dim,
        kernel_size=3,
        dilation_rate=2,
        padding="same",
        activation=activation
    )(x)

    x = layers.Dropout(dropout_rate)(x)

    # ==================================
    # 2. Patch Embedding
    # ==================================
    seq_len = input_shape[0]
    num_patches = seq_len // patch_size

    x = layers.Reshape(
        (num_patches, patch_size * projection_dim)
    )(x)

    x = layers.Dense(projection_dim)(x)

    # ==================================
    # 3. Learnable Positional Encoding
    # ==================================
    positions = tf.range(start=0, limit=num_patches, delta=1)

    pos_embedding = layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim
    )(positions)

    x = x + pos_embedding[tf.newaxis, ...]

    x = layers.Dropout(dropout_rate)(x)

    # ==================================
    # 4. Transformer Encoder Stack
    # ==================================
    for _ in range(num_transformer_blocks):
        x = transformer_block(
            x,
            num_heads=num_heads,
            projection_dim=projection_dim,
            ff_dim=ff_dim,
            activation=activation,
            dropout_rate=dropout_rate,
        )

    # ==================================
    # 5. Dual Pooling Head
    # ==================================
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)

    x = layers.Concatenate()([avg_pool, max_pool])

    x = layers.Dropout(dropout_rate)(x)

    # ==================================
    # 6. Regression Head
    # ==================================
    x = layers.Dense(64, activation=activation)(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)

    return model
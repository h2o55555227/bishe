import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ===============================
# Sin-Cos Positional Encoding
# ===============================
def get_sin_cos_pos_encoding(num_patches, projection_dim):
    """
    计算sin-cos位置编码
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    positions = tf.range(num_patches, dtype=tf.float32)
    positions = positions[:, tf.newaxis]  # (num_patches, 1)
    
    div_term = tf.exp(
        tf.range(0, projection_dim, 2, dtype=tf.float32) * 
        (-tf.math.log(10000.0) / projection_dim)
    )  # (projection_dim/2,)
    
    # 计算sin和cos部分
    pe_sin = tf.sin(positions * div_term)  # (num_patches, projection_dim/2)
    pe_cos = tf.cos(positions * div_term)  # (num_patches, projection_dim/2)
    
    # 交错拼接：sin, cos, sin, cos...
    pe = tf.stack([pe_sin, pe_cos], axis=-1)  # (num_patches, projection_dim/2, 2)
    pe = tf.reshape(pe, (num_patches, projection_dim))  # (num_patches, projection_dim)
    
    return pe


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
    # 1. CNN Local Feature Extractor (Upgraded)
    # ==================================
    # Conv1D + BatchNorm + GELU
    x = layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    
    # Conv1D
    x = layers.Conv1D(
        128,
        kernel_size=5,
        padding="same",
    )(x)
    
    # DepthwiseConv1D
    x = layers.DepthwiseConv1D(
        kernel_size=3,
        padding="same",
    )(x)
    
    # Final Conv1D to projection_dim
    x = layers.Conv1D(
        projection_dim,
        kernel_size=3,
        dilation_rate=2,
        padding="same",
        activation=activation,
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
    # 3. Sin-Cos Positional Encoding
    # ==================================
    pos_embedding = get_sin_cos_pos_encoding(num_patches, projection_dim)
    pos_embedding = tf.cast(pos_embedding, tf.float32)
    
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
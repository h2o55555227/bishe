import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


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
    projection_dim=192,
    num_heads=12,
    ff_dim=384,
    num_transformer_blocks=4,
    dropout_rate=0.12,
    patch_size=10,
    use_positional_encoding=True,
):
    inputs = keras.Input(shape=input_shape)
    
    # CNN局部特征提取 - 平衡版
    x = layers.Conv1D(filters=projection_dim, kernel_size=3, padding='same', activation=activation)(inputs)
    x = layers.Conv1D(filters=projection_dim, kernel_size=3, padding='same', activation=activation)(x)
    
    # Patch Embedding
    sequence_length, num_features = input_shape
    num_patches = sequence_length // patch_size
    
    # Reshape: (batch, sequence_length, features) -> (batch, num_patches, patch_size * features)
    x = layers.Reshape((num_patches, patch_size * projection_dim))(x)
    
    # Map to embedding dimension
    x = layers.Dense(projection_dim)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 位置编码
    if use_positional_encoding:
        position = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(position)
        x = x + position_embedding

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

    # Attention Pooling
    weights = layers.Dense(1)(x)
    weights = layers.Softmax(axis=1)(weights)
    x = layers.Multiply()([x, weights])
    x = layers.Lambda(lambda x: K.sum(x, axis=1), output_shape=(projection_dim,))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 预测头简化版
    x = layers.Dense(projection_dim // 2, activation=activation)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs)

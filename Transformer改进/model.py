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
    activation="swish",
    projection_dim=128,
    num_heads=8,
    ff_dim=256,
    num_transformer_blocks=2,
    dropout_rate=0.1,
    patch_size=12,
):
    inputs = keras.Input(shape=input_shape)
    
    # CNN局部特征提取
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
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs)

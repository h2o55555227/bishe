from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf


class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions


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
    activation="relu",
    projection_dim=128,
    num_heads=8,
    ff_dim=256,
    num_transformer_blocks=3,
    dropout_rate=0.1,
):
    inputs = keras.Input(shape=input_shape)
    
    sequence_length, num_features = input_shape
    
    x = layers.Dense(projection_dim)(inputs)
    x = layers.Dropout(dropout_rate)(x)
    
    x = PositionalEncoding(sequence_length, projection_dim)(x)

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

    weights = layers.Dense(1)(x)
    weights = layers.Softmax(axis=1)(weights)
    x = layers.Multiply()([x, weights])
    x = layers.Lambda(lambda x: K.sum(x, axis=1), output_shape=(projection_dim,))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs)

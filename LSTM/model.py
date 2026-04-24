from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(
    input_shape,
    lstm_units=128,
    dropout_rate=0.2,
    recurrent_dropout=0.1,
):
    """构建 LSTM 时间序列预测模型（单层，更快）"""
    inputs = keras.Input(shape=input_shape)
    
    # 单层 LSTM
    x = layers.LSTM(
        lstm_units,
        return_sequences=False,
        recurrent_dropout=recurrent_dropout,
    )(inputs)
    x = layers.Dropout(dropout_rate)(x)
    
    # 全连接层
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 输出层
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)

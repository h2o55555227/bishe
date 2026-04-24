from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(
    input_shape,
    lstm_units_1=128,
    lstm_units_2=64,
    dropout_rate=0.2,
    recurrent_dropout=0.1,
):
    """构建 LSTM 时间序列预测模型"""
    inputs = keras.Input(shape=input_shape)
    
    # 第一层 LSTM
    x = layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        recurrent_dropout=recurrent_dropout,
    )(inputs)
    x = layers.Dropout(dropout_rate)(x)
    
    # 第二层 LSTM
    x = layers.LSTM(
        lstm_units_2,
        return_sequences=False,
        recurrent_dropout=recurrent_dropout,
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 全连接层
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 输出层
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(
    input_shape,
    lstm_units=4,
    dropout_rate=0.3,
):
    """构建超简单的 LSTM 时间序列预测模型"""
    inputs = keras.Input(shape=input_shape)
    
    # 单层 LSTM，添加dropout
    x = layers.LSTM(lstm_units, return_sequences=False)(inputs)
    x = layers.Dropout(dropout_rate)(x)
    
    # 输出层
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)

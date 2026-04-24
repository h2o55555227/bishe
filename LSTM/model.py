from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(
    input_shape,
    lstm_units=32,
):
    """构建超简单的 LSTM 时间序列预测模型"""
    inputs = keras.Input(shape=input_shape)
    
    # 单层 LSTM，32 units，无 recurrent_dropout（更快）
    x = layers.LSTM(lstm_units, return_sequences=False)(inputs)
    
    # 输出层
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)

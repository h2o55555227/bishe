from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(
    input_shape,
    lstm_units=32,
):
    """构建 LSTM 时间序列预测模型，与 LSTM.ipynb 对齐"""
    inputs = keras.Input(shape=input_shape)
    
    # 单层 LSTM，32个单元
    lstm_out = layers.LSTM(lstm_units, return_sequences=False)(inputs)
    
    # 输出层
    outputs = layers.Dense(1)(lstm_out)
    
    return keras.Model(inputs, outputs)

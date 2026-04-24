import numpy as np


def compute_bias(true_values, predictions):
    """计算预测值相对于真实值的系统性偏差
    
    Args:
        true_values: 真实值数组
        predictions: 预测值数组
    
    Returns:
        bias: 平均偏差 (预测值 - 真实值)
    """
    true_values = np.array(true_values).flatten()
    predictions = np.array(predictions).flatten()
    bias = np.mean(predictions - true_values)
    return bias


def apply_bias_correction(predictions, bias):
    """对预测值应用偏差修正
    
    Args:
        predictions: 预测值数组
        bias: 需要修正的偏差值 (预测值 - 真实值)
    
    Returns:
        corrected_predictions: 修正后的预测值
    """
    predictions = np.array(predictions).flatten()
    corrected_predictions = predictions - bias
    return corrected_predictions


def bias_correction_pipeline(true_values, predictions):
    """完整的偏差修正流程：计算偏差 -> 修正预测 -> 返回修正结果
    
    Args:
        true_values: 真实值数组
        predictions: 预测值数组
    
    Returns:
        corrected_predictions: 修正后的预测值
        bias: 计算得到的偏差值
    """
    bias = compute_bias(true_values, predictions)
    corrected_predictions = apply_bias_correction(predictions, bias)
    return corrected_predictions, bias

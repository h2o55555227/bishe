import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


def compute_metrics(true_values, predictions):
    true_values = np.array(true_values).flatten()
    predictions = np.array(predictions).flatten()

    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    
    # SMAPE
    smape = 2 * np.mean(np.abs(true_values - predictions) / (np.abs(true_values) + np.abs(predictions)))
    
    # 额外统计
    true_mean = np.mean(true_values)
    pred_mean = np.mean(predictions)
    true_std = np.std(true_values)
    pred_std = np.std(predictions)
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape),
        "r2": float(r2),
        "true_mean": float(true_mean),
        "pred_mean": float(pred_mean),
        "true_std": float(true_std),
        "pred_std": float(pred_std),
    }

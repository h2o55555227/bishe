import numpy as np


def predict_arima_examples(
    model,
    val_data,
    train_data,
    num_examples=5,
    show_function=None,
    target_feature_index=1,
    future=72,
    step=6
):
    target_series = val_data.iloc[:, target_feature_index]
    train_target = train_data.iloc[:, target_feature_index]
    
    for i in range(min(num_examples, len(target_series) - future)):
        start_idx = i * step
        end_idx = start_idx + future
        
        if end_idx > len(target_series):
            break
        
        true_future = target_series.iloc[start_idx:end_idx].values
        
        combined_data = np.concatenate([train_target.values, target_series.iloc[:start_idx].values])
        from statsmodels.tsa.arima.model import ARIMA
        temp_model = ARIMA(combined_data, order=model.order, trend='n')
        temp_fit = temp_model.fit()
        model_pred = temp_fit.forecast(steps=future)
        
        print(f"--- 示例 {i + 1} ---")
        print(f"真实未来值 (前5个): {true_future[:5]}")
        print(f"模型预测 (前5个): {model_pred[:5]}")

        if show_function is not None:
            past_data = target_series.iloc[max(0, start_idx - 120):start_idx].values
            show_function(
                [past_data, true_future, model_pred],
                12,
                f"ARIMA 预测示例 {i + 1}",
            )


def predict_arima_all(
    model,
    val_data,
    train_data,
    target_feature_index=1,
    future=72,
    step=6,
    strategy="rolling"
):
    """
    ARIMA 预测函数，支持多种策略
    
    参数:
        strategy: 预测策略
            - "rolling": 滚动更新模型（默认，较准确但较慢）
            - "fixed": 固定模型，仅在训练集上训练一次（最快）
            - "hybrid": 混合策略，每 N 个窗口更新一次模型
    """
    target_series = val_data.iloc[:, target_feature_index]
    train_target = train_data.iloc[:, target_feature_index]
    
    all_true_values = []
    all_predictions = []
    
    num_predictions = (len(target_series) - future) // step
    
    if strategy == "fixed":
        from statsmodels.tsa.arima.model import ARIMA
        fixed_model = ARIMA(train_target.values, order=model.order, trend='n')
        fixed_fit = fixed_model.fit()
        
        for i in range(num_predictions):
            start_idx = i * step
            end_idx = start_idx + future
            
            true_future = target_series.iloc[start_idx:end_idx].values
            
            combined_data = np.concatenate([train_target.values, target_series.iloc[:start_idx].values])
            model_pred = fixed_fit.apply(combined_data).forecast(steps=future)
            
            all_true_values.extend(true_future)
            all_predictions.extend(model_pred)
    
    elif strategy == "hybrid":
        from statsmodels.tsa.arima.model import ARIMA
        update_interval = max(10, num_predictions // 10)
        current_fit = None
        
        for i in range(num_predictions):
            start_idx = i * step
            end_idx = start_idx + future
            
            true_future = target_series.iloc[start_idx:end_idx].values
            combined_data = np.concatenate([train_target.values, target_series.iloc[:start_idx].values])
            
            if i % update_interval == 0 or current_fit is None:
                temp_model = ARIMA(combined_data, order=model.order, trend='n')
                current_fit = temp_model.fit()
                model_pred = current_fit.forecast(steps=future)
            else:
                model_pred = current_fit.apply(combined_data).forecast(steps=future)
            
            all_true_values.extend(true_future)
            all_predictions.extend(model_pred)
    
    else:
        from statsmodels.tsa.arima.model import ARIMA
        for i in range(num_predictions):
            start_idx = i * step
            end_idx = start_idx + future
            
            true_future = target_series.iloc[start_idx:end_idx].values
            
            combined_data = np.concatenate([train_target.values, target_series.iloc[:start_idx].values])
            temp_model = ARIMA(combined_data, order=model.order, trend='n')
            temp_fit = temp_model.fit()
            model_pred = temp_fit.forecast(steps=future)
            
            all_true_values.extend(true_future)
            all_predictions.extend(model_pred)
    
    return all_true_values, all_predictions

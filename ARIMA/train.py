import os
import pickle


def train_arima_model(
    model,
    train_data,
    target_feature_index=1,
    trend='n',
    checkpoint_path="arima_model.pkl"
):
    target_series = train_data.iloc[:, target_feature_index]
    
    print(f"开始训练 ARIMA 模型，order={model.order}...")
    model.fit(target_series, trend=trend)
    print("ARIMA 模型训练完成")
    
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
        model.save(checkpoint_path)
        print(f"模型已保存至 {checkpoint_path}")
    
    return model

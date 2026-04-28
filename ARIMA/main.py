import csv
import json
import os

from data import (
    DATE_TIME_KEY,
    TARGET_FEATURE,
    colors,
    download_and_load_data,
    get_selected_features,
    normalize_features,
    selected_features,
    selected_titles,
    split_features,
)
from evaluate import compute_metrics
from model import build_arima_model
from predict import predict_arima_all, predict_arima_examples
from train import train_arima_model

from visualization import (
    show_comparison_visualization,
    show_processed_visualization,
    show_plot,
    show_raw_visualization,
    visualize_validation_predictions,
)


def save_predictions(true_values, predictions, output_dir=r"E:\毕设\实验\ARIMA\results", filename="predictions.csv"):
    import csv
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    
    with open(path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["真实值", "预测值"])
        writer.writerows(zip(true_values, predictions))
    
    print(f"预测结果已保存至 {path}")
    return path


def save_metrics(metrics, output_dir=r"E:\毕设\实验\ARIMA\results", filename="metrics.json"):
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    with open(path, mode="w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=4, ensure_ascii=False)
    
    print(f"指标已保存至 {path}")
    return path


def main():
    RESULTS_DIR = r"E:\毕设\实验\ARIMA\results"
    
    print("=== ARIMA 时间序列预测项目 ===")
    print("[1/7] 开始下载并加载数据...")
    df = download_and_load_data(data_dir=".")
    print(f"数据加载完成。数据集形状: {df.shape}")
    print(f"数据时间范围: {df[DATE_TIME_KEY].iloc[0]} 到 {df[DATE_TIME_KEY].iloc[-1]}")
    print(f"选择的特征: {selected_features}")
    print()

    print("[2/7] 开始选择特征...")
    raw_features = get_selected_features(df)
    input_feature_count = raw_features.shape[1]
    target_feature_index = raw_features.columns.get_loc(TARGET_FEATURE)
    print(f"特征选择完成。特征数据形状: {raw_features.shape}")
    print()

    train_split = int(0.715 * int(df.shape[0]))
    print(f"[3/7] 开始归一化处理和切分数据集... (训练集大小: {train_split})")
    normalized_features, train_mean, train_std = normalize_features(raw_features, train_split)
    print("归一化完成。")
    print(f"归一化参数 - 均值: {train_mean[:3]}..., 标准差: {train_std[:3]}...")
    print()

    train_data, val_data = split_features(normalized_features, train_split)
    print(f"训练数据形状: {train_data.shape}")
    print(f"验证数据形状: {val_data.shape}")
    print()

    print("[4/7] 开始构建 ARIMA 模型...")
    arima_order = (1, 1, 1)
    model_path = os.path.join(RESULTS_DIR, "arima_model.pkl")
    
    if os.path.exists(model_path):
        from model import ARIMAModel
        model = ARIMAModel.load(model_path)
        print(f"模型加载完成: {model_path}")
    else:
        model = build_arima_model(order=arima_order)
        model.train_mean = train_mean
        model.train_std = train_std
        print(f"模型构建完成，order={arima_order}")
    print()

    print("[5/7] 开始训练 ARIMA 模型...")
    if not os.path.exists(model_path):
        model = train_arima_model(
            model,
            train_data,
            target_feature_index=target_feature_index,
            trend='n',
            checkpoint_path=model_path
        )
    else:
        print("模型已存在，跳过训练")
    print()

    print("[6/7] 预测样例并可视化...")
    predict_arima_examples(
        model,
        val_data,
        train_data,
        num_examples=3,
        show_function=show_plot,
        target_feature_index=target_feature_index,
        future=72,
        step=6
    )
    print()

    print("[7/7] 开始对验证集进行预测并计算指标...")
    val_start_in_dataset = 52272
    val_start_idx = val_start_in_dataset - train_split
    all_true_values, all_predictions = predict_arima_all(
        model,
        val_data,
        train_data,
        target_feature_index=target_feature_index,
        future=72,
        step=6,
        strategy="fixed",
        start_idx=val_start_idx,
        num_points=400
    )
    metrics = compute_metrics(all_true_values, all_predictions)
    print("预测完成。")
    print(f"验证集大小: {len(all_true_values)}")
    print(f"验证 MAE: {metrics['mae']:.4f}")
    print(f"验证 MSE: {metrics['mse']:.4f}")
    print(f"验证 RMSE: {metrics['rmse']:.4f}")
    print(f"验证 MAPE: {metrics['mape']:.4f}")
    print(f"验证 SMAPE: {metrics['smape']:.4f}")
    print(f"验证 R2: {metrics['r2']:.4f}")
    print(f"真实值均值: {metrics['true_mean']:.4f}, 标准差: {metrics['true_std']:.4f}")
    print(f"预测值均值: {metrics['pred_mean']:.4f}, 标准差: {metrics['pred_std']:.4f}")
    print()
    
    print("[7.1/7] 开始可视化验证集预测结果...")
    visualize_validation_predictions(all_true_values, all_predictions, output_dir=RESULTS_DIR, start_idx=0, num_points=400)
    print("可视化完成。")
    print()

    save_predictions(all_true_values, all_predictions, output_dir=RESULTS_DIR)
    save_metrics(metrics, output_dir=RESULTS_DIR)
    
    print("所有结果已保存。执行完成！")


if __name__ == "__main__":
    main()

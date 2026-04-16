from data import (
    DATE_TIME_KEY,
    colors,
    download_and_load_data,
    get_selected_features,
    normalize_features,
    selected_features,
    selected_titles,
    split_features,
    build_timeseries_datasets,
)
from evaluate import compute_metrics
from model import build_transformer_model
from predict import predict_all, predict_examples
from train import train_model
import csv
import json
import os

from visualization import (
    show_comparison_visualization,
    show_processed_visualization,
    show_plot,
    show_raw_visualization,
    visualize_loss,
)

# 结果保存目录
RESULTS_DIR = "results"

# 手动挂载 Google Drive 说明：
# 在 Colab 中运行时，请先在一个单独的代码单元格中执行以下命令：
# 
# from google.colab import drive
# drive.mount('/content/drive')
# 
# 然后在本脚本中取消注释下面的代码行：
RESULTS_DIR = "/content/drive/MyDrive/transformer_project/results"

# 检查是否在 Colab 环境中
import sys
if 'google.colab' in sys.modules:
    print("检测到 Colab 环境")
    print("请确保已手动挂载 Google Drive")
    print("如果已挂载，请在脚本中设置 RESULTS_DIR 为 Drive 路径")
else:
    print("非 Colab 环境，使用本地路径")

print(f"结果将保存到: {RESULTS_DIR}")


def save_predictions(true_values, predictions, output_dir=RESULTS_DIR, filename="predictions.csv"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    with open(path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["真实值", "预测值"])
        writer.writerows(zip(true_values, predictions))

    print(f"预测结果已保存至 {path}")
    return path


def save_metrics(metrics, output_dir=RESULTS_DIR, filename="metrics.json"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    with open(path, mode="w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=4, ensure_ascii=False)

    print(f"指标已保存至 {path}")
    return path


def main():
    print("=== Transformer 时间序列预测项目 ===")
    print("[1/8] 开始下载并加载数据...")
    df = download_and_load_data(data_dir=".")
    print(f"数据加载完成。数据集形状: {df.shape}")
    print(f"数据时间范围: {df[DATE_TIME_KEY].iloc[0]} 到 {df[DATE_TIME_KEY].iloc[-1]}")
    print(f"选择的特征: {selected_features}")
    print()

    print("[2/8] 开始选择特征...")
    raw_features = get_selected_features(df)
    print(f"特征选择完成。特征数据形状: {raw_features.shape}")
    print()

    print("[2.1/8] 显示并保存原始数据可视化...")
    show_raw_visualization(df, selected_features, selected_titles, colors, output_dir=RESULTS_DIR)
    print()

    train_split = int(0.715 * int(df.shape[0]))
    print(f"[3/8] 开始归一化处理和切分数据集... (训练集大小: {train_split})")
    normalized_features, train_mean, train_std = normalize_features(raw_features, train_split)
    print("归一化完成。")
    print(f"归一化参数 - 均值: {train_mean[:3]}..., 标准差: {train_std[:3]}...")
    print()

    print("[3.1/8] 显示归一化后数据可视化...")
    show_processed_visualization(normalized_features, selected_titles, colors, output_dir=RESULTS_DIR)
    print()

    print("[3.2/8] 显示原始数据与归一化数据对比...")
    show_comparison_visualization(raw_features, normalized_features, selected_features, selected_titles, colors, output_dir=RESULTS_DIR)
    print()

    train_data, val_data = split_features(normalized_features, train_split)
    dataset_train, dataset_val, sequence_length = build_timeseries_datasets(
        train_data,
        val_data,
        normalized_features,
        train_split,
        past=720,
        future=72,
        step=6,
        batch_size=256,
    )
    print(f"[4/8] 时间序列数据集构建完成。序列长度: {sequence_length}")
    print(f"训练数据集大小: {len(dataset_train)} 批次")
    print(f"验证数据集大小: {len(dataset_val)} 批次")
    print()

    print("[5/8] 开始创建模型...")
    model = build_transformer_model((sequence_length, len(selected_features)), activation="relu")
    print("模型构建完成。")
    print("模型摘要:")
    model.summary()
    print()

    print("[6/8] 开始训练模型...")
    history = train_model(
        model,
        dataset_train,
        dataset_val,
        epochs=10,
        checkpoint_path="model_checkpoint.weights.h5",
        learning_rate=0.001,
    )
    print("训练完成。")
    print(f"最终训练损失: {history.history['loss'][-1]:.4f}")
    print(f"最终验证损失: {history.history['val_loss'][-1]:.4f}")
    print()

    print("[6.1/8] 可视化损失曲线...")
    visualize_loss(history, "Transformer + ReLU 损失曲线")
    print()

    print("[7/8] 预测样例并可视化...")
    predict_examples(model, dataset_val, num_examples=5, show_function=show_plot)
    print()

    print("[8/8] 开始对整个验证集进行预测并计算指标...")
    all_true_values, all_predictions = predict_all(model, dataset_val)
    metrics = compute_metrics(all_true_values, all_predictions)
    print("预测完成。")
    print(f"验证集大小: {len(all_true_values)}")
    print(f"验证 MAE: {metrics['mae']:.4f}")
    print(f"验证 MSE: {metrics['mse']:.4f}")
    print(f"验证 RMSE: {metrics['rmse']:.4f}")
    print(f"验证 MAPE: {metrics['mape']:.4f}")
    print(f"验证 SMAPE: {metrics['smape']:.4f}")
    print(f"验证 R²: {metrics['r2']:.4f}")
    print(f"真实值均值: {metrics['true_mean']:.4f}, 标准差: {metrics['true_std']:.4f}")
    print(f"预测值均值: {metrics['pred_mean']:.4f}, 标准差: {metrics['pred_std']:.4f}")
    print()

    save_predictions(all_true_values, all_predictions)
    save_metrics(metrics)
    
    # 保存完整模型
    model_save_path = os.path.join(RESULTS_DIR, "transformer_model.h5")
    model.save(model_save_path)
    print(f"完整模型已保存至: {model_save_path}")
    
    print("所有结果已保存。执行完成！")


if __name__ == "__main__":
    main()

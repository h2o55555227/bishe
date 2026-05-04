import csv
import json
import os
from tensorflow import keras

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
    build_timeseries_datasets,
)
from evaluate import compute_metrics
from model import build_transformer_model
from predict import predict_all, predict_examples
from train import train_model

from visualization import (
    show_comparison_visualization,
    show_processed_visualization,
    show_plot,
    show_raw_visualization,
    visualize_loss,
    visualize_validation_predictions,
)

# 消融实验模型配置
MODEL_CONFIGS = {
    "A": {
        "name": "原始Transformer",
        "use_cnn": False,
        "use_patch_embedding": False,
        "use_attention_pooling": False,
    },
    "B": {
        "name": "+CNN特征提取",
        "use_cnn": True,
        "use_patch_embedding": False,
        "use_attention_pooling": False,
    },
    "C": {
        "name": "+Patch Embedding",
        "use_cnn": True,
        "use_patch_embedding": True,
        "use_attention_pooling": False,
    },
    "D": {
        "name": "+Attention Pooling",
        "use_cnn": True,
        "use_patch_embedding": True,
        "use_attention_pooling": True,
    },
    "E": {
        "name": "完整模型",
        "use_cnn": True,
        "use_patch_embedding": True,
        "use_attention_pooling": True,
    },
}

# 选择要训练的模型（修改这里来切换 A-E）
SELECTED_MODEL = "B"


def save_predictions(true_values, predictions, output_dir=r"E:\毕设\实验\消融实验\results", filename="predictions.csv"):
    """保存预测结果到 CSV 文件"""
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


def save_metrics(metrics, output_dir=r"E:\毕设\实验\消融实验\results", filename="metrics.json"):
    """保存评估指标到 JSON 文件"""
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    with open(path, mode="w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=4, ensure_ascii=False)
    
    print(f"指标已保存至 {path}")
    return path


def main():
    # 定义结果保存目录，按模型编号区分
    RESULTS_DIR = rf"E:\毕设\实验\消融实验\results\model_{SELECTED_MODEL}"
    
    model_config = MODEL_CONFIGS[SELECTED_MODEL]
    print(f"=== 消融实验 - 模型 {SELECTED_MODEL}: {model_config['name']} ===")
    
    print("[1/8] 开始下载并加载数据...")
    df = download_and_load_data(data_dir=".")
    print(f"数据加载完成。数据集形状: {df.shape}")
    print(f"数据时间范围: {df[DATE_TIME_KEY].iloc[0]} 到 {df[DATE_TIME_KEY].iloc[-1]}")
    print(f"选择的特征: {selected_features}")
    print()

    print("[2/8] 开始选择特征...")
    raw_features = get_selected_features(df)
    input_feature_count = raw_features.shape[1]
    target_feature_index = raw_features.columns.get_loc(TARGET_FEATURE)
    print(f"特征选择完成。特征数据形状: {raw_features.shape}")
    print()

    train_split = int(0.715 * int(df.shape[0]))
    print(f"[3/8] 开始归一化处理和切分数据集... (训练集大小: {train_split})")
    normalized_features, train_mean, train_std = normalize_features(raw_features, train_split)
    print("归一化完成。")
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

    print(f"[5/8] 开始构建模型 {SELECTED_MODEL}: {model_config['name']}...")
    model = build_transformer_model(
        (sequence_length, input_feature_count),
        activation="swish",
        projection_dim=128,
        num_heads=4,
        ff_dim=512,
        num_transformer_blocks=3,
        dropout_rate=0.05,
        patch_size=4,
        use_cnn=model_config["use_cnn"],
        use_patch_embedding=model_config["use_patch_embedding"],
        use_attention_pooling=model_config["use_attention_pooling"],
    )
    print("模型构建完成。")
    print("模型摘要:")
    model.summary()
    print()

    print("[7/8] 预测样例并可视化...")
    predict_examples(model, dataset_val, num_examples=5, show_function=show_plot, target_feature_index=target_feature_index)
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
    print(f"验证 R2: {metrics['r2']:.4f}")
    print(f"真实值均值: {metrics['true_mean']:.4f}, 标准差: {metrics['true_std']:.4f}")
    print(f"预测值均值: {metrics['pred_mean']:.4f}, 标准差: {metrics['pred_std']:.4f}")
    print()
    
    print("[8.1/8] 开始可视化验证集预测结果...")
    visualize_validation_predictions(all_true_values, all_predictions, output_dir=RESULTS_DIR)
    print("可视化完成。")
    print()

    save_predictions(all_true_values, all_predictions, output_dir=RESULTS_DIR)
    save_metrics(metrics, output_dir=RESULTS_DIR)
    
    # 保存模型配置
    with open(os.path.join(RESULTS_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_id": SELECTED_MODEL,
            "model_name": model_config["name"],
            "config": model_config,
        }, f, indent=4, ensure_ascii=False)
    
    print(f"所有结果已保存至: {RESULTS_DIR}")
    print("执行完成！")


if __name__ == "__main__":
    main()

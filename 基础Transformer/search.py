import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def find_best_prediction_segment(predictions_df, window_size=100):
    true_values = predictions_df['true_value'].values
    predicted_values = predictions_df['predicted_value'].values
    
    errors = np.abs(true_values - predicted_values)
    
    best_start_idx = 0
    best_mae = float('inf')
    
    for i in range(len(errors) - window_size + 1):
        window_errors = errors[i:i+window_size]
        current_mae = np.mean(window_errors)
        if current_mae < best_mae:
            best_mae = current_mae
            best_start_idx = i
    
    best_end_idx = best_start_idx + window_size
    
    return best_start_idx, best_end_idx, best_mae


def inverse_normalize(normalized_values, mean, std):
    return normalized_values * std + mean


def visualize_best_segment(best_start_idx, best_end_idx, predictions_df, norm_params, output_dir='results'):
    true_values = predictions_df['true_value'].values
    predicted_values = predictions_df['predicted_value'].values
    
    temp_mean = norm_params['train_mean']['T (degC)']
    temp_std = norm_params['train_std']['T (degC)']
    
    true_temp = inverse_normalize(true_values, temp_mean, temp_std)
    pred_temp = inverse_normalize(predicted_values, temp_mean, temp_std)
    
    best_true = true_temp[best_start_idx:best_end_idx]
    best_pred = pred_temp[best_start_idx:best_end_idx]
    
    plt.figure(figsize=(16, 7))
    
    plt.plot(best_true, label='真实温度 (°C)', color='blue', linewidth=2)
    plt.plot(best_pred, label='预测温度 (°C)', color='red', linewidth=2, alpha=0.8)
    
    mae = np.mean(np.abs(best_true - best_pred))
    rmse = np.sqrt(np.mean((best_true - best_pred) ** 2))
    r2 = 1 - np.sum((best_true - best_pred) ** 2) / np.sum((best_true - np.mean(best_true)) ** 2)
    
    plt.title(f'预测最好的一段 (索引: {best_start_idx}-{best_end_idx})\nMAE: {mae:.4f}°C, RMSE: {rmse:.4f}°C, R2: {r2:.4f}', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('温度 (°C)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'best_prediction_segment.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f'最佳预测段可视化图已保存至 {save_path}')
    plt.show()
    
    plt.figure(figsize=(16, 5))
    residuals = best_true - best_pred
    plt.plot(residuals, label='残差 (真实-预测)', color='green', linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.8)
    plt.title('最佳预测段残差图', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('残差 (°C)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    residual_path = os.path.join(output_dir, 'best_segment_residuals.png')
    plt.savefig(residual_path, bbox_inches='tight', dpi=150)
    print(f'最佳预测段残差图已保存至 {residual_path}')
    plt.show()
    
    return best_start_idx, best_end_idx, mae, rmse, r2


def visualize_full_comparison(predictions_df, norm_params, output_dir='results'):
    true_values = predictions_df['true_value'].values
    predicted_values = predictions_df['predicted_value'].values
    
    temp_mean = norm_params['train_mean']['T (degC)']
    temp_std = norm_params['train_std']['T (degC)']
    
    true_temp = inverse_normalize(true_values, temp_mean, temp_std)
    pred_temp = inverse_normalize(predicted_values, temp_mean, temp_std)
    
    plt.figure(figsize=(16, 6))
    plt.plot(true_temp, label='真实温度', color='blue', alpha=0.7, linewidth=1)
    plt.plot(pred_temp, label='预测温度', color='red', alpha=0.7, linewidth=1)
    plt.title('完整预测结果对比', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('温度 (°C)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'full_prediction_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    print(f'完整预测对比图已保存至 {save_path}')
    plt.show()


def main():
    predictions_path = r'E:\毕设\实验\results\transformer终稿-20260426T105554Z-3-001\transformer终稿\predictions.csv'
    norm_params_path = r'E:\毕设\实验\results\transformer终稿-20260426T105554Z-3-001\transformer终稿\normalization_params.json'
    config_path = r'E:\毕设\实验\results\transformer终稿-20260426T105554Z-3-001\transformer终稿\run_config.json'
    output_dir = r'E:\毕设\实验\results\transformer终稿-20260426T105554Z-3-001\transformer终稿'
    
    predictions_df = pd.read_csv(predictions_path)
    print(f'预测数据加载完成，共 {len(predictions_df)} 个样本')
    
    with open(norm_params_path, 'r', encoding='utf-8') as f:
        norm_params = json.load(f)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    train_split = config['train_split']
    past = config['past']
    future = config['future']
    label_start = train_split + past + future
    
    window_size = 400
    best_start, best_end, best_mae = find_best_prediction_segment(predictions_df, window_size)
    
    raw_start = label_start + best_start
    raw_end = label_start + best_end - 1
    
    print(f'\n找到预测最好的一段:')
    print(f'  预测数据索引: {best_start} - {best_end - 1}')
    print(f'  原始数据索引: {raw_start} - {raw_end}')
    print(f'  窗口大小: {window_size}')
    print(f'  归一化MAE: {best_mae:.6f}')
    
    best_start_idx, best_end_idx, mae, rmse, r2 = visualize_best_segment(
        best_start, best_end, predictions_df, norm_params, output_dir
    )
    
    print(f'\n反归一化后的指标:')
    print(f'  MAE: {mae:.4f}°C')
    print(f'  RMSE: {rmse:.4f}°C')
    print(f'  R2: {r2:.4f}')
    
    print(f'\n数据说明:')
    print(f'  训练集大小: {train_split}')
    print(f'  past: {past}, future: {future}')
    print(f'  预测数据第0个样本对应原始数据第 {label_start} 个样本')
    print(f'  最佳预测段对应原始数据: 第 {raw_start} - {raw_end} 行')
    
    visualize_full_comparison(predictions_df, norm_params, output_dir)


if __name__ == '__main__':
    main()

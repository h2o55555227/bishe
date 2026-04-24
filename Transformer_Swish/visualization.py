import matplotlib.pyplot as plt
import pandas as pd
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def show_raw_visualization(
    data,
    feature_keys,
    titles,
    colors,
    date_time_key="Date Time",
    output_dir="results",
    filename="raw_data_visualization.png",
):
    time_data = pd.to_datetime(
        data[date_time_key], format="%d.%m.%Y %H:%M:%S", errors="raise"
    )
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20), dpi=80)

    for i, key in enumerate(feature_keys):
        c = colors[i % len(colors)]
        series = data[key].copy()
        series.index = time_data
        ax = series.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title=f"{titles[i]} - {key}（原始数据）",
            rot=25,
        )
        ax.set_xlabel("时间")
        ax.set_ylabel("数值")
        ax.legend([titles[i]])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"原始数据可视化图已保存至 {save_path}")
    plt.show()
    return save_path


def show_processed_visualization(features_df, selected_titles, colors, output_dir="results", filename="processed_data_visualization.png"):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), dpi=80)
    
    # 为时间特征添加标题
    time_titles = ["小时正弦", "小时余弦", "日正弦", "日余弦"]
    all_titles = selected_titles + time_titles
    
    for i, column in enumerate(features_df.columns):
        ax = axes[i // 3, i % 3]
        ax.plot(features_df[column].values, color=colors[i % len(colors)])
        if i < len(all_titles):
            ax.set_title(f"{all_titles[i]}（归一化）")
        else:
            ax.set_title(f"{column}（归一化）")
        ax.set_xlabel("时间步")
        ax.set_ylabel("归一化值")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"归一化数据可视化图已保存至 {save_path}")
    plt.show()
    return save_path


def show_comparison_visualization(raw_df, normalized_df, selected_features, selected_titles, colors, output_dir="results", filename="comparison_visualization.png"):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20), dpi=80)

    # 只对原始特征进行对比（不包含时间特征）
    original_features = [f for f in selected_features if f in raw_df.columns]
    
    for i, key in enumerate(original_features):
        if i >= 8:  # 限制在4行2列的布局内
            break
        ax = axes[i // 2, i % 2]
        ax.plot(raw_df[key].values, color=colors[i % len(colors)], label="原始数据")
        ax.plot(normalized_df[key].values, color="black", label="归一化数据")
        if i < len(selected_titles):
            ax.set_title(f"{selected_titles[i]} - 原始数据 vs 归一化数据")
        else:
            ax.set_title(f"{key} - 原始数据 vs 归一化数据")
        ax.set_xlabel("时间步")
        ax.set_ylabel("数值")
        ax.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"原始与归一化数据对比可视化图已保存至 {save_path}")
    plt.show()
    return save_path


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, "b", label="训练损失")
    plt.plot(epochs, val_loss, "r", label="验证损失")
    plt.title(title)
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.legend()
    plt.show()


def show_plot(plot_data, delta, title):
    labels = ["历史数据", "真实未来值", "模型预测"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-plot_data[0].shape[0], 0))
    future = delta or 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i == 0:
            plt.plot(time_steps, val.flatten(), marker[i], label=labels[i])
        else:
            plt.plot(future, val, marker[i], markersize=10, label=labels[i])

    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("时间步")
    plt.ylabel("数值")
    plt.show()


def visualize_validation_predictions(true_values, predictions, output_dir="results", filename="validation_predictions.png", num_points=400):
    """可视化整个验证集的真实值和预测值对比"""
    import numpy as np
    
    # 转换为 NumPy 数组并只取前 num_points 个点
    true_values = np.array(true_values)[:num_points]
    predictions = np.array(predictions)[:num_points]
    
    # 保存主图
    plt.figure(figsize=(16, 6))
    plt.plot(true_values, label="真实值", color="blue", alpha=0.7, linewidth=1.5)
    plt.plot(predictions, label="预测值", color="red", alpha=0.7, linewidth=1.5)
    plt.title(f"验证集真实值 vs 预测值对比 (前{num_points}个点)")
    plt.xlabel("样本索引")
    plt.ylabel("数值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    print(f"验证集预测对比图已保存至 {save_path}")
    plt.show()
    
    # 绘制残差图
    plt.figure(figsize=(16, 4))
    residuals = true_values - predictions
    plt.plot(residuals, label="残差", color="green", alpha=0.7)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.8)
    plt.title(f"残差图（真实值 - 预测值）(前{num_points}个点)")
    plt.xlabel("样本索引")
    plt.ylabel("残差")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存残差图
    residual_filename = "validation_residuals.png"
    residual_path = os.path.join(output_dir, residual_filename)
    plt.savefig(residual_path, bbox_inches="tight", dpi=100)
    print(f"残差图已保存至 {residual_path}")
    plt.show()
    
    return save_path

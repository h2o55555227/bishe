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
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12), dpi=80)

    for i, column in enumerate(features_df.columns):
        ax = axes[i // 2, i % 2]
        ax.plot(features_df[column].values, color=colors[i % len(colors)])
        ax.set_title(f"{selected_titles[i]}（归一化）")
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

    for i, key in enumerate(selected_features):
        ax = axes[i // 2, i % 2]
        ax.plot(raw_df[key].values, color=colors[i % len(colors)], label="原始数据")
        ax.plot(normalized_df[key].values, color="black", label="归一化数据")
        ax.set_title(f"{selected_titles[i]} - 原始数据 vs 归一化数据")
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

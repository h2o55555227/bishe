
import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from cnn_model import build_cnn_model

CNN_DIR = Path(__file__).parent


def generate_keras_plot_model():
    """使用keras自带的plot_model生成结构图"""
    print("[1/3] 构建CNN模型...")
    
    # 计算输入形状: past=240, step=6, feature_count=7+4=11
    sequence_length = int(240 / 6)  # 40
    feature_count = 11
    
    model = build_cnn_model(
        input_shape=(sequence_length, feature_count),
        activation="relu",
        filters=[32, 64],
        kernel_size=3,
        dropout_rate=0.15,
        use_batch_norm=False,
        residual_connection=False,
    )
    
    print("模型构建完成！")
    model.summary()
    
    # 尝试使用plot_model（需要graphviz）
    try:
        print("\n[2/3] 尝试使用plot_model生成结构图...")
        from tensorflow.keras.utils import plot_model
        
        plot_path = CNN_DIR / "cnn_model_architecture.png"
        plot_model(
            model,
            to_file=str(plot_path),
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=150,
        )
        print(f"Keras plot_model结构图已保存至: {plot_path}")
        return True
    except Exception as e:
        print(f"plot_model失败 (可能缺少graphviz): {e}")
        return False


def generate_text_diagram():
    """生成文本形式的结构图"""
    print("\n[3/3] 生成文本结构图...")
    
    diagram = """
╔══════════════════════════════════════════════════════════════════════════╗
║                          CNN 模型结构示意图                                ║
╚══════════════════════════════════════════════════════════════════════════╝

输入: (batch_size, 40, 11)
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        CNN Block 1 (32 filters)                           │
├──────────────────────────────────────────────────────────────────────────┤
│  Conv1D(filters=32, kernel_size=3, padding='same')  →  (40, 32)           │
│  Activation('relu')                                                      │
│  MaxPooling1D(pool_size=2)                     →  (20, 32)               │
│  Dropout(0.15)                                                            │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        CNN Block 2 (64 filters)                           │
├──────────────────────────────────────────────────────────────────────────┤
│  Conv1D(filters=64, kernel_size=3, padding='same')  →  (20, 64)           │
│  Activation('relu')                                                      │
│  MaxPooling1D(pool_size=2)                     →  (10, 64)               │
│  Dropout(0.15)                                                            │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        额外卷积层                                          │
├──────────────────────────────────────────────────────────────────────────┤
│  Conv1D(filters=64, kernel_size=3, padding='same')  →  (10, 64)           │
│  Activation('relu')                                                      │
│  Dropout(0.15)                                                            │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        全局平均池化 + 全连接层                             │
├──────────────────────────────────────────────────────────────────────────┤
│  GlobalAveragePooling1D()                  →  (64,)                      │
│  Dropout(0.15)                                                            │
│  Dense(64, activation='relu')               →  (64,)                      │
│  Dropout(0.15)                                                            │
│  Dense(1)                                    →  (1,)                       │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
输出: (batch_size, 1)

╔══════════════════════════════════════════════════════════════════════════╗
║                              配置参数                                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  输入序列长度: 40 (240步 / 步长6)                                          ║
║  特征维度: 11 (7个气象特征 + 4个时间特征)                                   ║
║  卷积核大小: 3                                                             ║
║  Dropout率: 0.15                                                           ║
║  激活函数: ReLU                                                            ║
║  损失函数: MAE                                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    
    text_path = CNN_DIR / "cnn_model_structure.txt"
    text_path.write_text(diagram, encoding="utf-8")
    print(f"文本结构图已保存至: {text_path}")
    
    print("\n" + "="*80)
    print(diagram)
    print("="*80)


def generate_matplotlib_diagram():
    """使用matplotlib生成可视化结构图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        
        print("\n[额外] 尝试使用matplotlib生成可视化图...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # 现代渐变色系 - 柔和的蓝紫色调
# 统一高级蓝紫渐变色系（同一色系深浅变化）
        colors = {
            'input':   '#312E81',   # 最深靛蓝
            'conv1':   '#4338CA',   # 深蓝紫
            'conv2':   '#6366F1',   # 主蓝紫
            'conv3':   '#818CF8',   # 中浅蓝紫
            'pool':    '#A5B4FC',   # 浅蓝紫
            'dropout': '#C7D2FE',   # 极浅蓝紫
            'dense':   '#E0E7FF',   # 淡雾蓝
            'output':  '#EEF2FF'    # 最浅收尾色
        }
        
        # 绘制各层
        layers = [
            {'name': 'Input\n(40, 11)', 'y': 18, 'color': colors['input'], 'text_color': 'white'},
            {'name': 'Conv1D\n(32, 3x3)', 'y': 16, 'color': colors['conv1'], 'text_color': 'white'},
            {'name': 'ReLU', 'y': 15, 'color': colors['conv1'], 'text_color': 'white'},
            {'name': 'MaxPool1D\n(2x)', 'y': 14, 'color': colors['pool'], 'text_color': '#333'},
            {'name': 'Dropout\n(0.15)', 'y': 13, 'color': colors['dropout'], 'text_color': '#333'},
            {'name': 'Conv1D\n(64, 3x3)', 'y': 11, 'color': colors['conv2'], 'text_color': 'white'},
            {'name': 'ReLU', 'y': 10, 'color': colors['conv2'], 'text_color': 'white'},
            {'name': 'MaxPool1D\n(2x)', 'y': 9, 'color': colors['pool'], 'text_color': '#333'},
            {'name': 'Dropout\n(0.15)', 'y': 8, 'color': colors['dropout'], 'text_color': '#333'},
            {'name': 'Conv1D\n(64, 3x3)', 'y': 6, 'color': colors['conv3'], 'text_color': 'white'},
            {'name': 'ReLU', 'y': 5, 'color': colors['conv3'], 'text_color': 'white'},
            {'name': 'Dropout\n(0.15)', 'y': 4, 'color': colors['dropout'], 'text_color': '#333'},
            {'name': 'GlobalAvgPool', 'y': 3, 'color': colors['pool'], 'text_color': '#333'},
            {'name': 'Dense\n(64)', 'y': 2, 'color': colors['dense'], 'text_color': '#333'},
            {'name': 'Dense\n(1)', 'y': 1, 'color': colors['output'], 'text_color': '#333'},
        ]
        
        # 绘制箭头（使用渐变色）
        for i in range(len(layers)-1):
            ax.arrow(5, layers[i]['y'] - 0.3, 0, layers[i+1]['y'] - layers[i]['y'] + 0.6,
                    head_width=0.15, head_length=0.2, fc='#667eea', ec='#667eea', alpha=0.6)
        
        # 绘制层框（添加阴影效果）
        for layer in layers:
            # 阴影层
            shadow = mpatches.FancyBboxPatch(
                (3.05, layer['y'] - 0.45), 4, 0.8,
                boxstyle="round,pad=0.1",
                facecolor='gray',
                alpha=0.2
            )
            ax.add_patch(shadow)
            
            # 主层
            rect = mpatches.FancyBboxPatch(
                (3, layer['y'] - 0.4), 4, 0.8,
                boxstyle="round,pad=0.15",
                facecolor=layer['color'],
                alpha=0.95,
                edgecolor='white',
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(5, layer['y'], layer['name'],
                   ha='center', va='center', fontsize=12, fontweight='bold', 
                   color=layer['text_color'])
        
        # 添加标题（渐变色文字效果）
        title = 'CNN 温度预测模型架构'
        ax.text(5, 19.5, title,
               ha='center', va='center', fontsize=18, fontweight='bold',
               color='#667eea')
        
        plt_path = CNN_DIR / "cnn_model_visualization.png"
        plt.tight_layout()
        plt.savefig(plt_path, dpi=150, bbox_inches='tight')
        print(f"Matplotlib可视化图已保存至: {plt_path}")
        plt.close()
        
    except Exception as e:
        print(f"Matplotlib可视化失败: {e}")


def main():
    print("="*80)
    print("开始生成CNN模型结构图...")
    print("="*80)
    
    # 1. 尝试Keras plot_model
    keras_plot_success = generate_keras_plot_model()
    
    # 2. 生成文本结构图
    generate_text_diagram()
    
    # 3. 尝试matplotlib可视化
    generate_matplotlib_diagram()
    
    print("\n" + "="*80)
    print("结构图生成完成！")
    if keras_plot_success:
        print("✓ cnn_model_architecture.png (Keras plot_model)")
    print("✓ cnn_model_structure.txt (文本结构图)")
    print("✓ cnn_model_visualization.png (Matplotlib可视化)")
    print("="*80)


if __name__ == "__main__":
    main()


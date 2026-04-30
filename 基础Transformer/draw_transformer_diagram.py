import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 确保输出目录存在
output_dir = r"E:\毕设\实验\Transformer\results"
os.makedirs(output_dir, exist_ok=True)


def draw_transformer_architecture():
    fig, ax = plt.subplots(figsize=(24, 40), dpi=100)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 40)
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'input': '#4CAF50',
        'conv': '#2196F3',
        'patch': '#FF9800',
        'transformer': '#9C27B0',
        'attention': '#F44336',
        'output': '#00BCD4',
    }

    
    # ========== 输入层 ==========
    y = 35.5
    draw_box(ax, 4, y, 16, 2.5, '输入层', colors['input'], 
             '输入形状: (sequence_length, features)')
    
    # ========== CNN局部特征提取 ==========
    y = 31.0
    draw_box(ax, 4, y, 7.5, 2.2, 'Conv1D Layer 1', colors['conv'], 
             '过滤器: 128  卷积核: 3  激活函数: gelu  padding: same')
    draw_box(ax, 12.5, y, 7.5, 2.2, 'Conv1D Layer 2', colors['conv'], 
             '过滤器: 128  卷积核: 3  激活函数: gelu  padding: same')
    
    # ========== Patch Embedding ==========
    y = 26.5
    draw_box(ax, 4, y, 16, 2.5, 'Patch Embedding', colors['patch'], 
             'Patch大小: 12  Reshape → Dense(128) → Dropout(0.1)  将时间序列分割成patch并嵌入')
    
    # ========== Transformer Encoder ==========
    y_base = 22.0
    block_height = 3.5
    spacing = 1.0
    
    for i in range(3):
        y = y_base - i * (block_height + spacing)
        draw_transformer_block(ax, 4, y, 16, block_height, f'Transformer Block {i+1}', colors)
    
    # ========== Attention Pooling ==========
    y = 7.0
    draw_box(ax, 4, y, 16, 4.8, 'Attention Pooling', colors['attention'], 
             '步骤:\n1. Dense(1) + Softmax → 学习注意力权重\n2. Multiply → 对每个patch加权\n3. Sum(axis=1) → 加权求和\n替代: 简单取最后时间步')
    
    # ========== 输出层 ==========
    y = 3.0
    draw_box(ax, 4, y, 16, 2.5, '输出层', colors['output'], 
             'Dropout(0.1) → Dense(1)\n回归预测任务')
    
    # 连接箭头
    draw_vertical_arrow(ax, 12, 35.5, 33.2)  # Input底部 -> Conv1顶部
    draw_vertical_arrow(ax, 12, 31.0, 29.0)  # Conv底部 -> Patch顶部
    draw_vertical_arrow(ax, 12, 26.5, 25.5)  # Patch底部 -> Block1顶部
    draw_vertical_arrow(ax, 12, 22.0, 21.0)  # Block1底部 -> Block2顶部
    draw_vertical_arrow(ax, 12, 17.5, 16.5)  # Block2底部 -> Block3顶部
    draw_vertical_arrow(ax, 12, 13.0, 11.8)  # Block3底部 -> Attention顶部
    draw_vertical_arrow(ax, 12, 7.0, 5.5)   # Attention底部 -> 输出层顶部
    
    # 图例
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='输入层'),
        mpatches.Patch(color=colors['conv'], label='CNN层'),
        mpatches.Patch(color=colors['patch'], label='Patch Embedding'),
        mpatches.Patch(color=colors['transformer'], label='Transformer Block'),
        mpatches.Patch(color=colors['attention'], label='Attention Pooling'),
        mpatches.Patch(color=colors['output'], label='输出层'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'transformer_architecture_diagram.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"模型架构图已保存至: {save_path}")
    plt.show()


def draw_box(ax, x, y, width, height, title, color, description):
    # 绘制框
    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.08",
        facecolor=color, alpha=0.8, edgecolor=color, linewidth=2
    )
    ax.add_patch(rect)
    
    # 标题
    ax.text(x + width/2, y + height - 0.4, title, 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # 描述
    ax.text(x + width/2, y + height - 1.1, description, 
            ha='center', va='top', fontsize=8, color='white', linespacing=2.0)


def draw_transformer_block(ax, x, y, width, height, title, colors):
    # 绘制外框
    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.08",
        facecolor='white', edgecolor=colors['transformer'], linewidth=3, linestyle='--'
    )
    ax.add_patch(rect)
    
    # 标题
    ax.text(x + width/2, y + height - 0.3, title, 
            ha='center', va='top', fontsize=9, fontweight='bold', color=colors['transformer'])
    
    # 内部结构
    sub_x = x + 1.0
    sub_width = (width - 3.0) / 2
    sub_y = y + 0.6
    sub_height = height - 1.5
    
    # 多头注意力
    draw_sub_box(ax, sub_x, sub_y, sub_width, sub_height, 
                '多头自注意力  num_heads=8  key_dim=128\nLayerNorm + 残差连接', colors['transformer'])
    
    # 前馈网络
    draw_sub_box(ax, sub_x + sub_width + 1.0, sub_y, sub_width, sub_height, 
                '前馈网络  FFN: 256 → 128  激活: gelu\nLayerNorm + 残差连接', colors['transformer'])


def draw_sub_box(ax, x, y, width, height, title, color):
    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, title, 
            ha='center', va='center', fontsize=9, color=color, linespacing=1.8)


def draw_vertical_arrow(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', lw=3.5, color='#555555', shrinkA=12, shrinkB=12))


if __name__ == "__main__":
    draw_transformer_architecture()

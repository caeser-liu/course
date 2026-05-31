"""
训练过程可视化脚本
功能：
1. 记录训练指标（loss、每类准确率）
2. 绘制 loss 曲线
3. 绘制每类准确率变化曲线
4. 绘制彩色混淆矩阵
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 类别名称
CLASS_NAMES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc',
    'Random', 'Scratch', 'Near-full', 'None'
]

NUM_CLASSES = 9


class MetricsRecorder:
    """训练指标记录器，挂载到 train.py 的训练循环中使用"""

    def __init__(self, save_dir="./training_metrics"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = {
            'train_loss': [],
            'val_acc': [],
            'per_class_acc': [],  # 每个 epoch 的每类准确率列表
            'confusion_matrices': []  # 每个评估点的混淆矩阵
        }

    def record_epoch(self, epoch, train_loss, val_acc,
                     per_class_acc=None, conf_matrix=None):
        """记录一个 epoch 的指标"""
        self.history['train_loss'].append(train_loss)
        self.history['val_acc'].append(val_acc)
        self.history['per_class_acc'].append(per_class_acc)
        if conf_matrix is not None:
            self.history['confusion_matrices'].append(conf_matrix)

    def save(self):
        """保存历史记录到 JSON"""
        save_path = os.path.join(self.save_dir, 'metrics.json')
        # 将 numpy 数组转为 list 以便 JSON 序列化
        serializable = {
            'train_loss': self.history['train_loss'],
            'val_acc': self.history['val_acc'],
            'per_class_acc': [
                arr.tolist() if isinstance(arr, np.ndarray) else arr
                for arr in self.history['per_class_acc']
            ],
            'confusion_matrices': [
                cm.tolist() if isinstance(cm, np.ndarray) else cm
                for cm in self.history['confusion_matrices']
            ]
        }
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=4)
        print(f"指标已保存到 {save_path}")

    @classmethod
    def load(cls, save_dir="./training_metrics"):
        """从 JSON 加载历史记录"""
        recorder = cls(save_dir)
        save_path = os.path.join(save_dir, 'metrics.json')
        with open(save_path, 'r') as f:
            data = json.load(f)
        recorder.history['train_loss'] = data['train_loss']
        recorder.history['val_acc'] = data['val_acc']
        recorder.history['per_class_acc'] = data['per_class_acc']
        recorder.history['confusion_matrices'] = [
            np.array(cm) for cm in data['confusion_matrices']
        ]
        return recorder


# ============================================================
# 绘图函数
# ============================================================

def plot_loss_curve(history, save_path=None):
    """绘制训练 loss 曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], 'b-', linewidth=2, marker='o', markersize=4)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss 曲线已保存到 {save_path}")
    plt.show()


def plot_per_class_accuracy(history, save_path=None):
    """绘制每个类别的准确率变化曲线"""
    plt.figure(figsize=(12, 7))
    per_class = history['per_class_acc']
    epochs = range(1, len(per_class) + 1)

    for i in range(NUM_CLASSES):
        acc_per_epoch = [pc[i] for pc in per_class if pc is not None and len(pc) == NUM_CLASSES]
        if acc_per_epoch:
            plt.plot(epochs[:len(acc_per_epoch)], acc_per_epoch,
                    marker='o', markersize=4, linewidth=1.5, label=CLASS_NAMES[i])

    plt.title('Per-Class Accuracy Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"每类准确率曲线已保存到 {save_path}")
    plt.show()


def plot_overall_accuracy(history, save_path=None):
    """绘制总体验证准确率曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_acc'], 'r-', linewidth=2, marker='s', markersize=4)
    plt.title('Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"验证准确率曲线已保存到 {save_path}")
    plt.show()


def plot_combined_curves(history, save_path=None):
    """将 Loss 和 Val Acc 画在同一图（双 Y 轴）"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    line1 = ax1.plot(history['train_loss'], color=color, linewidth=2, marker='o', markersize=4, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Val Accuracy', color=color, fontsize=12)
    line2 = ax2.plot(history['val_acc'], color=color, linewidth=2, marker='s', markersize=4, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title('Training Loss & Validation Accuracy', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"组合曲线已保存到 {save_path}")
    plt.show()


def plot_confusion_matrix(conf_matrix, save_path=None, title='Confusion Matrix'):
    """绘制彩色混淆矩阵"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 归一化（按真实标签，即每行之和为1）
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_norm = np.where(row_sums > 0, conf_matrix / row_sums, 0)

    im = ax.imshow(conf_matrix_norm, cmap='Blues', aspect='auto')

    # 颜色条
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Proportion', rotation=-90, va="bottom", fontsize=11)

    # 刻度和标签
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)

    # 在格子中写数值（原始数量 + 百分比）
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            count = conf_matrix[i, j]
            pct = conf_matrix_norm[i, j]
            if count > 0:
                color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{count}\n({pct:.1%})",
                       ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存到 {save_path}")
    plt.show()


def plot_all(history, output_dir="./training_metrics"):
    """一键生成所有图表"""
    os.makedirs(output_dir, exist_ok=True)

    plot_loss_curve(history, os.path.join(output_dir, 'loss_curve.png'))
    plot_overall_accuracy(history, os.path.join(output_dir, 'val_accuracy.png'))
    plot_combined_curves(history, os.path.join(output_dir, 'loss_and_accuracy.png'))
    plot_per_class_accuracy(history, os.path.join(output_dir, 'per_class_accuracy.png'))

    # 如果有混淆矩阵，绘制最后一个（最佳模型对应的）
    if history['confusion_matrices']:
        last_cm = history['confusion_matrices'][-1]
        plot_confusion_matrix(
            last_cm,
            os.path.join(output_dir, 'confusion_matrix.png'),
            title='Confusion Matrix (Best Model)'
        )

    print(f"\n所有图表已保存到 {output_dir}/")


# ============================================================
# 独立运行模式：从已有 metrics.json 绘图
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='训练过程可视化')
    parser.add_argument('--metrics_dir', type=str, default='./training_metrics',
                       help='metrics.json 所在目录')
    parser.add_argument('--conf_matrix', type=str, default=None,
                       help='混淆矩阵 JSON/CSV 文件路径（可选）')
    args = parser.parse_args()

    # 加载训练指标
    recorder = MetricsRecorder.load(args.metrics_dir)
    history = recorder.history

    print(f"加载了 {len(history['train_loss'])} 个 epoch 的记录")
    print(f"最终 val_acc: {history['val_acc'][-1]:.4f}")

    # 绘图
    plot_all(history, output_dir=args.metrics_dir)

    # 如果有单独的混淆矩阵文件
    if args.conf_matrix and os.path.exists(args.conf_matrix):
        if args.conf_matrix.endswith('.json'):
            with open(args.conf_matrix, 'r') as f:
                cm = np.array(json.load(f))
        elif args.conf_matrix.endswith('.npy'):
            cm = np.load(args.conf_matrix)
        else:
            cm = np.loadtxt(args.conf_matrix, delimiter=',', dtype=int)
        plot_confusion_matrix(cm, title='Confusion Matrix (from file)')


if __name__ == '__main__':
    main()

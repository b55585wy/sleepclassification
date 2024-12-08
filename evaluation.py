import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def f1_scores_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    使用混淆矩阵计算每个类别的 F1 分数

    :param cm: 混淆矩阵，形状为 (num_classes, num_classes)

    :return: 每个类别的 F1 分数，形状为 (num_classes,)
    """
    # 计算真正例 (True Positives), 实际为正例的总数 (Rel), 预测为正例的总数 (Sel)
    tp = np.diagonal(cm).astype(np.float32)
    rel = np.sum(cm, axis=0).astype(np.float32)  # 实际为正例的总数 (Recall denominator)
    sel = np.sum(cm, axis=1).astype(np.float32)  # 预测为正例的总数 (Precision denominator)

    # 计算精确率 (Precision) 和召回率 (Recall)
    precision = np.divide(tp, sel, out=np.zeros_like(tp), where=sel > 0)
    recall = np.divide(tp, rel, out=np.zeros_like(tp), where=rel > 0)

    # 计算 F1 分数
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return f1

def plot_confusion_matrix(cm: np.ndarray, classes: list,
                          normalize: bool = True, title: str = None,
                          cmap: str = 'Blues', path: str = ''):
    """
    绘制混淆矩阵并保存为图像文件

    :param cm: 混淆矩阵，形状为 (num_classes, num_classes)
    :param classes: 每个类别的名称列表
    :param normalize: 是否归一化混淆矩阵
    :param title: 图表标题
    :param cmap: 颜色映射
    :param path: 图像保存路径
    """
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix, without normalization'

    # 复制混淆矩阵以避免修改原始数据
    cm_to_plot = cm.astype('float') if normalize else cm.copy()

    if normalize:
        cm_to_plot /= cm_to_plot.sum(axis=1)[:, np.newaxis]
        cm_to_plot = np.nan_to_num(cm_to_plot)  # 将 NaN 替换为 0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_to_plot, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Normalized Count' if normalize else 'Count', rotation=-90, va="bottom")

    # 设置标签
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 在每个单元格中添加文本注释
    fmt = '.2f' if normalize else 'd'
    thresh = cm_to_plot.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm_to_plot[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_to_plot[i, j] > thresh else "black")

    fig.tight_layout()

    # 确保保存路径存在
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # 规范化文件名，移除可能导致路径错误的字符
    safe_title = title.replace(" ", "_").replace(",", "").replace("'", "").replace("-", "_")
    save_path = os.path.join(path, f"{safe_title}.png")
    try:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to '{save_path}'.")
    except Exception as e:
        logging.error(f"Error saving confusion matrix to '{save_path}': {e}")
    finally:
        plt.close(fig)  # 关闭图形以释放内存

def draw_training_plot(history: list, from_fold: int, train_folds: int, output_path: str):
    """
    绘制训练和验证的准确率与损失曲线，并保存为图像文件

    :param history: 每个折叠的训练历史，列表中每个元素是一个字典，包含 'accuracy', 'val_accuracy', 'loss', 'val_loss'
    :param from_fold: 起始折叠编号
    :param train_folds: 训练的折叠数量
    :param output_path: 图像保存目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # 创建图形，调整 figsize 以适应折叠数量
    fig_height = 6 * train_folds if train_folds <= 10 else 60  # 限制最大高度
    fig, axes = plt.subplots(train_folds, 2, figsize=(15, fig_height))
    if train_folds == 1:
        axes = np.expand_dims(axes, axis=0)  # 确保 axes 是二维数组

    for i in range(train_folds):
        fold_history = history[i]
        # 处理不同版本的 Keras 指标名称
        acc = fold_history.get('accuracy') or fold_history.get('acc')
        val_acc = fold_history.get('val_accuracy') or fold_history.get('val_acc')
        loss = fold_history.get('loss')
        val_loss = fold_history.get('val_loss')

        epochs = range(1, len(acc) + 1)

        # 绘制准确率
        ax_acc = axes[i, 0] if train_folds > 1 else axes[0]
        ax_acc.plot(epochs, acc, 'C0-', label='Training Accuracy')
        ax_acc.plot(epochs, val_acc, 'C1-.', label='Validation Accuracy')
        ax_acc.set_title(f'Training and Validation Accuracy in Fold {from_fold + i}')
        ax_acc.set_xlabel('Epochs')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)

        # 绘制损失
        ax_loss = axes[i, 1] if train_folds > 1 else axes[1]
        ax_loss.plot(epochs, loss, 'C0-', label='Training Loss')
        ax_loss.plot(epochs, val_loss, 'C1-.', label='Validation Loss')
        ax_loss.set_title(f'Training and Validation Loss in Fold {from_fold + i}')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

    plt.tight_layout()
    save_filename = f"f{from_fold}-{from_fold + train_folds - 1}_accuracy_and_loss.png"
    save_path = os.path.join(output_path, save_filename)
    try:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Training plots saved to '{save_path}'.")
    except Exception as e:
        logging.error(f"Error saving training plots to '{save_path}': {e}")
    finally:
        plt.close(fig)  # 关闭图形以释放内存

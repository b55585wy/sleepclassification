import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_evaluation_results():
    # 准备数据
    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    f1_scores = [0.7391, 0.5283, 0.9219, 0.7436, 0.9006]
    
    # 1. F1分数条形图
    plt.figure(figsize=(10, 6))
    plt.bar(classes, f1_scores, color='skyblue')
    plt.title('F1 Scores by Sleep Stage')
    plt.xlabel('Sleep Stage')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.savefig('evaluation_results/f1_scores.png')
    plt.close()
    
    # 2. 混淆矩阵热力图
    confusion_matrix = np.array([
        [17, 6, 2, 0, 1],
        [2, 14, 6, 0, 8],
        [0, 2, 313, 0, 3],
        [1, 0, 19, 29, 0],
        [0, 1, 21, 0, 154]
    ])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('evaluation_results/confusion_matrix.png')
    plt.close()
    
    # 3. 性能指标雷达图
    metrics = {
        'W': {'precision': 0.85, 'recall': 0.65, 'f1': 0.7391},
        'N1': {'precision': 0.61, 'recall': 0.47, 'f1': 0.5283},
        'N2': {'precision': 0.87, 'recall': 0.98, 'f1': 0.9219},
        'N3': {'precision': 1.00, 'recall': 0.59, 'f1': 0.7436},
        'REM': {'precision': 0.93, 'recall': 0.88, 'f1': 0.9006}
    }
    
    # 创建雷达图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2*np.pi, len(classes), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
    
    for metric in ['precision', 'recall', 'f1']:
        values = [metrics[c][metric] for c in classes]
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, '-o', label=metric.capitalize())
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Performance Metrics by Sleep Stage')
    plt.savefig('evaluation_results/radar_plot.png')
    plt.close()

# 执行绘图
plot_evaluation_results()
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from evaluation import evaluate_model, draw_training_plot
from load_files import load_npz_files
from preprocess import prepare_data
import yaml

def load_training_history(log_dir):
    """加载TensorBoard日志中的训练历史"""
    from tensorflow.python.summary.summary_iterator import summary_iterator
    
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    
    for e in summary_iterator(tf.io.gfile.glob(f"{log_dir}/*")[0]):
        for v in e.summary.value:
            if v.tag in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
                history[v.tag].append(v.simple_value)
    
    return history

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载配置
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    # 加载最新的模型
    model_path = 'checkpoints/two_stream_salient_best.h5'  # 根据实际路径修改
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path)
    logging.info(f"Loaded model from {model_path}")
    
    # 加载数据
    data_dir = 'data'  # 根据实际路径修改
    data_list, labels_list = load_npz_files(data_dir)
    
    # 准备数据
    (train_eeg, train_eog, train_labels), (val_eeg, val_eog, val_labels) = \
        prepare_data(data_list, labels_list, 
                    sequence_length=params['preprocess']['sequence_epochs'])
    
    # 评估模型
    logging.info("Evaluating model...")
    eval_results = evaluate_model(
        model=model,
        test_data=[val_eeg, val_eog],
        test_labels=val_labels,
        class_names=['W', 'N1', 'N2', 'N3', 'REM']
    )
    
    # 加载并绘制训练历史
    logging.info("Loading training history...")
    latest_log_dir = sorted(tf.io.gfile.glob('logs/tensorboard/*'))[-1]
    history = load_training_history(latest_log_dir)
    
    logging.info("Generating training plots...")
    draw_training_plot(
        history=[history],
        from_fold=0,
        train_folds=1,
        output_path='evaluation_results'
    )
    
    # 打印评估结果摘要
    print("\nEvaluation Summary:")
    print(f"Average F1 Score: {np.mean(eval_results['f1_scores']):.4f}")
    print("\nPer-class F1 Scores:")
    for i, class_name in enumerate(['W', 'N1', 'N2', 'N3', 'REM']):
        print(f"{class_name}: {eval_results['f1_scores'][i]:.4f}")

if __name__ == "__main__":
    main() 
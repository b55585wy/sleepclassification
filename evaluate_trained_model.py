import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
from evaluation import evaluate_model, draw_training_plot
from load_files import load_npz_files
from preprocess import prepare_data
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_training_history(log_dir):
    """加载TensorBoard日志中的训练历史"""
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    
    try:
        # 查找事件文件
        event_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    event_files.append(os.path.join(root, file))
        
        if not event_files:
            logging.error(f"No event files found in {log_dir}")
            return None
            
        logging.info(f"Loading history from {event_files[0]}")
        
        # 加载事件文件
        event_acc = EventAccumulator(event_files[0])
        event_acc.Reload()
        
        # 读取标量数据
        tags = event_acc.Tags()['scalars']
        
        if 'accuracy' in tags:
            history['accuracy'] = [s.value for s in event_acc.Scalars('accuracy')]
        if 'val_accuracy' in tags:
            history['val_accuracy'] = [s.value for s in event_acc.Scalars('val_accuracy')]
        if 'loss' in tags:
            history['loss'] = [s.value for s in event_acc.Scalars('loss')]
        if 'val_loss' in tags:
            history['val_loss'] = [s.value for s in event_acc.Scalars('val_loss')]
        
        logging.info(f"Loaded history with {len(history['accuracy'])} epochs")
        return history
        
    except Exception as e:
        logging.error(f"Error loading training history: {e}")
        return None

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载配置
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    # 加载最新的模型
    model_path = 'checkpoints/two_stream_salient_best.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path)
    logging.info(f"Loaded model from {model_path}")
    
    # 加载数据
    data_dir = 'data'
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
    try:
        log_dirs = tf.io.gfile.glob('logs/tensorboard/*')
        if not log_dirs:
            logging.error("No tensorboard logs found")
            return
            
        latest_log_dir = sorted(log_dirs)[-1]
        logging.info(f"Using latest log dir: {latest_log_dir}")
        
        history = load_training_history(latest_log_dir)
        if history:
            logging.info("Generating training plots...")
            draw_training_plot(
                history=[history],
                from_fold=0,
                train_folds=1,
                output_path='evaluation_results'
            )
            logging.info("Training plots saved successfully")
        else:
            logging.error("Failed to load training history")
            
    except Exception as e:
        logging.error(f"Error processing training history: {e}")
    
    # 打印评估结果摘要
    print("\nEvaluation Summary:")
    print(f"Average F1 Score: {np.mean(eval_results['f1_scores']):.4f}")
    print("\nPer-class F1 Scores:")
    for i, class_name in enumerate(['W', 'N1', 'N2', 'N3', 'REM']):
        print(f"{class_name}: {eval_results['f1_scores'][i]:.4f}")

if __name__ == "__main__":
    main() 
import os
import yaml
import logging
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from load_files import load_npz_files
from preprocess import prepare_data
from model import TwoSteamSalientModelWrapper

def setup_logging():
    """设置日志"""
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def create_callbacks(model_name):
    """创建回调函数"""
    # 创建checkpoints目录
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    callbacks = [
        # 模型检查点
        ModelCheckpoint(
            f'checkpoints/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # 早停
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率调整
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard
        TensorBoard(
            log_dir=f'logs/tensorboard/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            histogram_freq=1
        )
    ]
    return callbacks

def train(args):
    """训练模型"""
    # 检测可用的GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 允许GPU内存增长，防止占用所有内存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPUs: {gpus}")
            
            # 创建分布式策略
            if len(gpus) > 1:
                strategy = tf.distribute.MirroredStrategy()
                print(f'Training using {strategy.num_replicas_in_sync} GPUs')
            else:
                strategy = tf.distribute.get_strategy()  # 默认策略
                print('Training using single GPU')
        except RuntimeError as e:
            print(f"GPU error: {e}")
    else:
        strategy = tf.distribute.get_strategy()  # CPU策略
        print('No GPUs found, using CPU')

    # 加载配置
    logging.info("Loading configuration from %s", args.config)
    with open(args.config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    # 加载数据
    logging.info("Loading data from %s", args.data_dir)
    data_list, labels_list = load_npz_files(args.data_dir)
    
    # 使用策略范围创建模型
    with strategy.scope():
        # 准备数据
        logging.info("Preparing data...")
        (train_eeg, train_eog, train_labels), (val_eeg, val_eog, val_labels) = \
            prepare_data(data_list, labels_list, 
                        sequence_length=params['preprocess']['sequence_epochs'],
                        test_mode=args.test_mode)
        
        logging.info("Data shapes:")
        logging.info(f"Train EEG: {train_eeg.shape}")
        logging.info(f"Train EOG: {train_eog.shape}")
        logging.info(f"Train Labels: {train_labels.shape}")
        logging.info(f"Val EEG: {val_eeg.shape}")
        logging.info(f"Val EOG: {val_eog.shape}")
        logging.info(f"Val Labels: {val_labels.shape}")
        
        # 创建模型
        logging.info("Creating model...")
        model_wrapper = TwoSteamSalientModelWrapper(params)
        model = model_wrapper.model
        
        # 配置优化器
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params['train'].get('learning_rate', 0.001)
        )
        
        # 编译模型
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # 设置批次大小
    batch_size = params['train']['batch_size'] * strategy.num_replicas_in_sync
    
    # 创建回调函数
    callbacks = create_callbacks(params['model_name'])
    
    # 训练模型
    logging.info("Starting training...")
    try:
        history = model.fit(
            [train_eeg, train_eog],  # 输入
            train_labels,            # 标签
            batch_size=batch_size,
            epochs=params['train']['epochs'],
            validation_data=([val_eeg, val_eog], val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        logging.info("Training completed successfully!")
        return history
        
    except Exception as e:
        logging.error("Error during training: %s", str(e))
        raise

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./sleepedf/prepared',
                      help='Directory containing the data files')
    parser.add_argument('--config', type=str, default='hyperparameters.yaml',
                      help='Path to configuration file')
    parser.add_argument('--test_mode', action='store_true',
                      help='Run in test mode with limited data')
    args = parser.parse_args()
    
    try:
        # 训练模型
        train_history = train(args)
        logging.info("Process completed successfully!")
        
    except Exception as e:
        logging.error("Process failed: %s", str(e))
        raise

if __name__ == '__main__':
    main()

import os
import yaml
import logging
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from load_files import load_npz_files
from preprocess import prepare_data
from model import TwoSteamSalientModelWrapper
from data_generator import SleepDataGenerator

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
    # 加载配置
    logging.info("Loading configuration from %s", args.config)
    with open(args.config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    # 加载数据
    logging.info("Loading data from %s", args.data_dir)
    data_files = load_npz_files(args.data_dir)
    
    # 准备数据
    logging.info("Preparing data...")
    (train_eeg, train_eog, train_labels), (val_eeg, val_eog, val_labels) = \
        prepare_data(data_files, 
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
    
    # 编译模型
    model.compile(
        optimizer=params['train']['optimizer'],
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 创建回调函数
    callbacks = create_callbacks(params['model_name'])
    
    # 创建数据生成器
    train_generator = SleepDataGenerator(
        data_files=data_files[:-5],  # 除了最后5个文件
        batch_size=params['train']['batch_size'],
        sequence_length=params['preprocess']['sequence_epochs']
    )
    
    # 创建验证生成器
    val_generator = SleepDataGenerator(
        data_files=data_files[-5:],  # 最后5个文件
        batch_size=params['train']['batch_size'],
        sequence_length=params['preprocess']['sequence_epochs'],
        shuffle=False
    )
    
    # 训练模型
    logging.info("Starting training...")
    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=params['train']['epochs'],
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=True,
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

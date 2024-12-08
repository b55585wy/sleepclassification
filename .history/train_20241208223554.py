import os
import glob
import logging
import argparse
import itertools
from functools import reduce

import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from preprocess import preprocess
from load_files import load_npz_files
from loss_function import weighted_categorical_cross_entropy
from evaluation import f1_scores_from_cm, plot_confusion_matrix
from model import SingleSalientModel, TwoSteamSalientModel



def parse_args():
    """
    Parse command line arguments and validate them.
    """
    parser = argparse.ArgumentParser(description="Evaluate Sleep Models with K-Fold Cross-Validation.")

    parser.add_argument("--data_dir", "-d", default="./sleepedf/prepared",
                        help="Directory where the data is stored.")
    parser.add_argument("--modal", '-m', type=int, choices=[0, 1], default=1,
                        help="Modal: 0 for single modal, 1 for multi modal.")
    parser.add_argument("--output_dir", '-o', default='./result',
                        help="Directory to save the results.")
    parser.add_argument("--valid", '-v', type=int, default=20,
                        help="Number of folds for k-fold validation.")

    args = parser.parse_args()

    # Validate k_folds
    if args.valid <= 0:
        parser.error("The `valid` argument should be a positive integer.")

    return args


def configure_logging():
    """
    Configure the logging settings.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler("evaluation.log", mode='w')
                        ])


def load_hyperparameters(hyperparams_path="hyperparameters.yaml"):
    """
    Load hyperparameters from a YAML file.
    """
    if not os.path.exists(hyperparams_path):
        logging.critical(f"Hyperparameters file '{hyperparams_path}' not found.")
        exit(-1)

    with open(hyperparams_path, encoding='utf-8') as f:
        hyper_params = yaml.safe_load(f)

    return hyper_params


def build_and_compile_model(modal, hyper_params):
    """
    Instantiate and compile the model based on the modality.
    """
    if modal == 0:
        model = SingleSalientModel(**hyper_params)
    else:
        model = TwoSteamSalientModel(**hyper_params)

    # Ensure 'class_weights' is in hyper_params for the loss function
    if 'class_weights' not in hyper_params:
        logging.critical("Missing 'class_weights' in hyperparameters.")
        exit(-1)

    loss = weighted_categorical_cross_entropy(hyper_params['class_weights'])

    model.compile(optimizer=hyper_params.get('optimizer', 'adam'),
                  loss=loss,
                  metrics=['accuracy'])

    return model


def evaluate_model_on_fold(model, model_path, test_data, test_labels, modal, batch_size):
    """
    Load model weights and evaluate on the test data for a single fold.
    """
    # Load weights
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_path}' not found.")
        return None, None, None

    model.load_weights(model_path)
    logging.info(f"Loaded weights from '{model_path}'.")

    # Predict
    if modal == 0:
        y_pred = model.predict(test_data, batch_size=batch_size)
    else:
        y_pred = model.predict(test_data, batch_size=batch_size)

    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    test_labels = test_labels.reshape((-1, test_labels.shape[-1]))

    # Compute metrics
    acc = accuracy_score(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
    f1 = f1_score(test_labels.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    cm = confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))

    logging.info(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    logging.info("Confusion Matrix:")
    logging.info(f"{cm.astype('float32') / cm.sum():.4f}")

    return acc, f1, cm


def train(args, hyper_params):
    """
    训练模型的主函数
    
    :param args: 命令行参数
    :param hyper_params: 超参数字典
    :return: 训练历史
    """
    # 初始化结果列表
    acc_list = []
    val_acc_list = []
    loss_list = []
    val_loss_list = []

    # 加载数据
    print("Loading data...")
    data = load_npz_files(args.data_dir)
    
    # 数据预处理
    print("Preprocessing data...")
    x_train, y_train = preprocess(data, hyper_params)
    
    # 构建模型
    print("Building model...")
    if args.modal == 0:
        model = SingleSalientModel(**hyper_params)
    else:
        model = TwoSteamSalientModel(**hyper_params)
    
    # 编译模型
    print("Compiling model...")
    loss = weighted_categorical_cross_entropy(hyper_params['class_weights'])
    model.compile(optimizer=hyper_params.get('optimizer', 'adam'),
                 loss=loss,
                 metrics=['accuracy'])
    
    # 设置回调函数
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10
        )
    ]
    
    # 开始训练
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=hyper_params['train']['epochs'],
        batch_size=hyper_params['train']['batch_size'],
        validation_split=0.2,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 记录训练历史
    acc_list.append(history.history['accuracy'])
    val_acc_list.append(history.history['val_accuracy'])
    loss_list.append(history.history['loss'])
    val_loss_list.append(history.history['val_loss'])
    
    # 返回训练历史
    return {
        'acc': acc_list,
        'val_acc': val_acc_list,
        'loss': loss_list,
        'val_loss': val_loss_list
    }


def main():
    try:
        # 解析参数
        print("Parsing arguments...")
        args = parse_args()
        
        # 配置日志
        print("Configuring logging...")
        configure_logging()
        
        # 加载超参数
        print("Loading hyperparameters...")
        hyper_params = load_hyperparameters()
        
        # 检查数据目录
        print(f"Checking data directory: {args.data_dir}")
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory {args.data_dir} does not exist!")
            return
            
        # 获取数据文件列表
        npz_files = glob.glob(os.path.join(args.data_dir, "*.npz"))
        print(f"Found {len(npz_files)} npz files")
        
        # 确保输出目录存在
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 创建或加载split文件
        split_path = os.path.join(args.output_dir, "split.npz")
        if not os.path.exists(split_path):
            print("Creating new split file...")
            # 这里添加创建split文件的代码
        else:
            print("Loading existing split file...")
            
        # ���始训练
        print("Starting training...")
        train_history = train(args, hyper_params)
        
        print("Training completed!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

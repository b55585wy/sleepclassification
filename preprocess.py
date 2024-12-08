import numpy as np
from typing import Tuple, List

def create_sequences(data: np.ndarray, labels: np.ndarray, 
                    sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    将数据组织成序列
    
    Args:
        data: 原始数据，形状 (n_samples, 3000, 3)
        labels: 原始标签，形状 (n_samples,) 或 (n_samples, num_classes)
        sequence_length: 序列长度，默认20
    
    Returns:
        序列数据和对应的标签
        - 数据形状: (n_sequences, sequence_length, 3000, 3)
        - 标签形状: (n_sequences, sequence_length, num_classes)
    """
    # 使用float32而不是float64
    data = data.astype(np.float32)
    
    n_samples = len(data)
    n_sequences = n_samples - sequence_length + 1
    
    # 创建序列，指定dtype=float32
    sequences = np.zeros((n_sequences, sequence_length, *data.shape[1:]), dtype=np.float32)
    
    # 滑动窗口创建序列
    for i in range(n_sequences):
        sequences[i] = data[i:i + sequence_length]
    
    # 处理标签
    if len(labels.shape) == 1:  # 如果标签是一维的
        sequence_labels = np.zeros((n_sequences, sequence_length))
        for i in range(n_sequences):
            sequence_labels[i] = labels[i:i + sequence_length]
        # 转换标签为one-hot编码
        num_classes = len(np.unique(labels))
        sequence_labels = np.eye(num_classes)[sequence_labels.astype(int)]
    else:  # 如果标签已经是one-hot编码
        sequence_labels = np.zeros((n_sequences, sequence_length, labels.shape[1]))
        for i in range(n_sequences):
            sequence_labels[i] = labels[i:i + sequence_length]
    
    return sequences, sequence_labels

def prepare_data(data_list: List[np.ndarray], labels_list: List[np.ndarray], 
                sequence_length: int = 20, test_mode: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    准备训练和验证数据
    
    Args:
        data_list: 数据列表，每个元素形状 (n_samples, 3000, 3)
        labels_list: 标签列表，每个元素形状 (n_samples,)
        sequence_length: 序列长度，默认20
        test_mode: 是否为测试模式，如果是则只处理少量数据
    """
    all_sequences = []
    all_labels = []
    
    # 在测试模式下只处理前3个文件
    if test_mode:
        print("Running in test mode - processing only first 3 files")
        data_list = data_list[:3]
        labels_list = labels_list[:3]
    
    for idx, (data, labels) in enumerate(zip(data_list, labels_list)):
        try:
            print(f"\nProcessing file {idx}")
            print(f"Data shape: {data.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Unique labels: {np.unique(labels)}")
            
            # 转换为float32
            data = data.astype(np.float32)
            
            # 分离EEG和EOG数据
            eeg_data = data[..., :3]
            eog_data = data[..., -3:]
            
            # 创建序列
            eeg_sequences, sequence_labels = create_sequences(eeg_data, labels, sequence_length)
            eog_sequences, _ = create_sequences(eog_data, labels, sequence_length)
            
            print(f"Created sequences - EEG shape: {eeg_sequences.shape}")
            print(f"Labels shape: {sequence_labels.shape}")
            
            all_sequences.append((eeg_sequences, eog_sequences))
            all_labels.append(sequence_labels)
            
        except Exception as e:
            print(f"Error processing file {idx}: {str(e)}")
            continue
    
    if not all_sequences:
        raise ValueError("No sequences were created!")
    
    # 合并数据
    eeg_data = np.concatenate([seq[0] for seq in all_sequences])
    eog_data = np.concatenate([seq[1] for seq in all_sequences])
    labels = np.concatenate(all_labels)
    
    print("\nFinal data shapes:")
    print(f"EEG: {eeg_data.shape}")
    print(f"EOG: {eog_data.shape}")
    print(f"Labels: {labels.shape}")
    
    # 分割训练集和验证集
    split_idx = int(len(eeg_data) * 0.8)
    
    return (eeg_data[:split_idx], eog_data[:split_idx], labels[:split_idx]), \
           (eeg_data[split_idx:], eog_data[split_idx:], labels[split_idx:])

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    标准化数据
    
    Args:
        data: 输入数据，形状 (n_sequences, sequence_length, 3000, 3)
    
    Returns:
        标准化后的数据
    """
    # 在时间维度上计算均值和标准差
    mean = np.mean(data, axis=(1, 2), keepdims=True)
    std = np.std(data, axis=(1, 2), keepdims=True)
    
    # 标准化
    normalized_data = (data - mean) / (std + 1e-8)
    
    return normalized_data

if __name__ == "__main__":
    # 测试代码
    from load_files import load_npz_files
    import yaml
    
    # 加载配置
    with open("hyperparameters.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # 加载数据
    data_list, labels_list = load_npz_files("./sleepedf/prepared")
    
    # 准备数据
    (train_eeg, train_eog, train_labels), (val_eeg, val_eog, val_labels) = \
        prepare_data(data_list, labels_list, sequence_length=params['preprocess']['sequence_epochs'])
    
    # 标准化数据
    train_eeg = normalize_data(train_eeg)
    train_eog = normalize_data(train_eog)
    val_eeg = normalize_data(val_eeg)
    val_eog = normalize_data(val_eog)
    
    # 打印数据形状
    print(f"Training data shapes:")
    print(f"EEG: {train_eeg.shape}")
    print(f"EOG: {train_eog.shape}")
    print(f"Labels: {train_labels.shape}")
    
    print(f"\nValidation data shapes:")
    print(f"EEG: {val_eeg.shape}")
    print(f"EOG: {val_eog.shape}")
    print(f"Labels: {val_labels.shape}")

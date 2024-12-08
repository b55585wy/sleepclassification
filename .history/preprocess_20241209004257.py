import numpy as np
from typing import Tuple, List

def create_sequences(data: np.ndarray, labels: np.ndarray, 
                    sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    将数据组织成序列
    
    Args:
        data: 原始数据，形状 (n_samples, 3000, 3)
        labels: 原始标签，形状 (n_samples,)
        sequence_length: 序列长度，默认20
    
    Returns:
        序列数据和对应的标签
        - 数据形状: (n_sequences, sequence_length, 3000, 3)
        - 标签形状: (n_sequences, sequence_length, num_classes)
    """
    n_samples = len(data)
    n_sequences = n_samples - sequence_length + 1
    
    # 创建序列
    sequences = np.zeros((n_sequences, sequence_length, *data.shape[1:]))
    sequence_labels = np.zeros((n_sequences, sequence_length))
    
    # 滑动窗口创建序列
    for i in range(n_sequences):
        sequences[i] = data[i:i + sequence_length]
        sequence_labels[i] = labels[i:i + sequence_length]
    
    # 转换标签为one-hot编码
    num_classes = len(np.unique(labels))
    one_hot_labels = np.eye(num_classes)[sequence_labels.astype(int)]
    
    return sequences, one_hot_labels

def prepare_data(data_list: List[np.ndarray], labels_list: List[np.ndarray], 
                sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    准备训练和验证数据
    
    Args:
        data_list: 数据列表，每个元素形状 (n_samples, 3000, 3)
        labels_list: 标签列表，每个元素形状 (n_samples,)
        sequence_length: 序列长度，默认20
    
    Returns:
        eeg_data, eog_data, labels, 每个都分为训练集和验证集
    """
    all_sequences = []
    all_labels = []
    
    # 处理每个文件的数据
    for data, labels in zip(data_list, labels_list):
        # 分离EEG和EOG数据
        eeg_data = data[..., :3]  # 假设前3个通道是EEG
        eog_data = data[..., -3:]  # 假设后3个通道是EOG
        
        # 创建序列
        eeg_sequences, labels = create_sequences(eeg_data, labels, sequence_length)
        eog_sequences, _ = create_sequences(eog_data, labels, sequence_length)
        
        all_sequences.append((eeg_sequences, eog_sequences))
        all_labels.append(labels)
    
    # 合并所有数据
    eeg_data = np.concatenate([seq[0] for seq in all_sequences])
    eog_data = np.concatenate([seq[1] for seq in all_sequences])
    labels = np.concatenate(all_labels)
    
    # 打乱数据
    indices = np.random.permutation(len(eeg_data))
    eeg_data = eeg_data[indices]
    eog_data = eog_data[indices]
    labels = labels[indices]
    
    # 分割训练集和验证集
    split_idx = int(len(eeg_data) * 0.8)  # 80% 训练，20% 验证
    
    train_eeg = eeg_data[:split_idx]
    train_eog = eog_data[:split_idx]
    train_labels = labels[:split_idx]
    
    val_eeg = eeg_data[split_idx:]
    val_eog = eog_data[split_idx:]
    val_labels = labels[split_idx:]
    
    return (train_eeg, train_eog, train_labels), (val_eeg, val_eog, val_labels)

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
    data_list, labels_list = load_npz_files("./data")
    
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

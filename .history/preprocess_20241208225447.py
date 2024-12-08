import numpy as np
import logging


def normalization(data: np.ndarray) -> np.ndarray:
    """
    Normalize the PSG data by subtracting the mean and dividing by the standard deviation.

    :param data: PSG data array of shape [num_samples, epoch_len, channels]
    :return: Normalized PSG data
    """
    mean = np.mean(data, axis=(1, 2), keepdims=True)
    std = np.std(data, axis=(1, 2), keepdims=True) + 1e-8  # Prevent division by zero
    return (data - mean) / std


def data_big_group(d: np.ndarray, big_group_size: int) -> np.ndarray:
    """
    Divide data into big groups to prevent data leak in data enhancement.

    :param d: PSG data array of shape [num_samples, epoch_len, channels]
    :param big_group_size: Size of each big group
    :return: Big grouped data of shape [num_samples, num_big_groups, big_group_size, channels]
    """
    num_big_groups = d.shape[1] // big_group_size
    if num_big_groups == 0:
        raise ValueError("big_group_size is larger than epoch_len.")
    d = d[:, :num_big_groups * big_group_size, :]
    return d.reshape(d.shape[0], num_big_groups, big_group_size, d.shape[2])


def label_big_group(l: np.ndarray, big_group_size: int) -> np.ndarray:
    """
    Divide labels into big groups to prevent data leak in data enhancement.

    :param l: Labels array of shape [num_samples, epoch_len]
    :param big_group_size: Size of each big group
    :return: Big grouped labels of shape [num_samples, num_big_groups, big_group_size]
    """
    num_big_groups = l.shape[1] // big_group_size
    if num_big_groups == 0:
        raise ValueError("big_group_size is larger than the length of labels.")
    l = l[:, :num_big_groups * big_group_size]
    return l.reshape(l.shape[0], num_big_groups, big_group_size)


def data_window_slice(d: np.ndarray, sequence_epochs: int, stride: int) -> np.ndarray:
    """
    Slice data into windows for data enhancement.

    :param d: Big grouped data of shape [num_samples, num_big_groups, big_group_size, channels]
    :param sequence_epochs: Number of epochs in each window
    :param stride: Stride for window sliding
    :return: Window sliced data of shape [num_samples, num_big_groups, num_windows, sequence_epochs, channels]
    """
    num_samples, num_big_groups, big_group_size, channels = d.shape
    num_windows = (big_group_size - sequence_epochs) // stride + 1
    if num_windows <= 0:
        raise ValueError("sequence_epochs is larger than big_group_size.")

    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + sequence_epochs
        window = d[:, :, start:end, :]  # Shape: [num_samples, num_big_groups, sequence_epochs, channels]
        windows.append(window)
    return np.stack(windows, axis=3)  # Shape: [num_samples, num_big_groups, num_windows, sequence_epochs, channels]


def labels_window_slice(l: np.ndarray, sequence_epochs: int, stride: int) -> np.ndarray:
    """
    Slice labels into windows for data enhancement.

    :param l: Big grouped labels of shape [num_samples, num_big_groups, big_group_size]
    :param sequence_epochs: Number of epochs in each window
    :param stride: Stride for window sliding
    :return: Window sliced labels of shape [num_samples, num_big_groups, num_windows, sequence_epochs]
    """
    num_samples, num_big_groups, big_group_size = l.shape
    num_windows = (big_group_size - sequence_epochs) // stride + 1
    if num_windows <= 0:
        raise ValueError("sequence_epochs is larger than big_group_size.")

    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + sequence_epochs
        window = l[:, :, start:end]  # Shape: [num_samples, num_big_groups, sequence_epochs]
        windows.append(window)
    return np.stack(windows, axis=3)  # Shape: [num_samples, num_big_groups, num_windows, sequence_epochs]


def preprocess(data, labels, param):
    """
    预处理数据
    
    :param data: 原始数据列表
    :param labels: 标签列表
    :param param: 预处理参数字典
    :return: 预处理后的数据和标签
    """
    print("Preprocessing data with parameters:", param)
    
    sequence_epochs = param['sequence_epochs']
    print(f"Sequence length: {sequence_epochs}")
    
    processed_data = []
    processed_labels = []
    
    # 添加调试信息
    print(f"Input data length: {len(data)}")
    print(f"Input labels length: {len(labels)}")
    
    for idx, (d, l) in enumerate(zip(data, labels)):
        try:
            # 打印每个数据的形状
            print(f"Processing file {idx}, data shape: {d.shape}, labels shape: {l.shape}")
            
            # 确保数据长度是sequence_epochs的整数倍
            n_samples = len(d)
            n_sequences = n_samples // sequence_epochs
            if n_sequences == 0:
                print(f"Skipping file {idx}: too short ({n_samples} < {sequence_epochs})")
                continue
                
            # 裁剪数据
            d = d[:n_sequences * sequence_epochs]
            l = l[:n_sequences * sequence_epochs]
            
            # 重塑数据
            d = d.reshape(-1, sequence_epochs, *d.shape[1:])
            l = l.reshape(-1, sequence_epochs)
            
            print(f"After reshape - data shape: {d.shape}, labels shape: {l.shape}")
            
            processed_data.append(d)
            processed_labels.append(l)
            
        except Exception as e:
            print(f"Error processing file {idx}: {str(e)}")
            continue
    
    if not processed_data:
        raise ValueError("No data was successfully processed!")
    
    # 合并所有数据
    x = np.concatenate(processed_data, axis=0)
    y = np.concatenate(processed_labels, axis=0)
    
    print(f"Final preprocessed data shape: {x.shape}")
    print(f"Final preprocessed labels shape: {y.shape}")
    
    return x, y


if __name__ == "__main__":
    # 测试代码
    import yaml
    from load_files import load_npz_files
    
    # 加载配置
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        hyper_params = yaml.full_load(f)
    
    # 加载测试数据
    test_data = np.random.rand(1000, 3000, 1, 1)  # 示例数据
    test_labels = np.random.randint(0, 5, (1000,))  # 示例标签
    
    # 测试预处理
    try:
        x, y = preprocess([test_data], [test_labels], hyper_params['preprocess'])
        print("Test successful!")
        print(f"Output shapes: x={x.shape}, y={y.shape}")
    except Exception as e:
        print(f"Test failed: {str(e)}")

import numpy as np


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
    
    for d, l in zip(data, labels):
        # 确保数据长度是sequence_epochs的整数倍
        n_samples = len(d)
        n_sequences = n_samples // sequence_epochs
    Preprocess the raw PSG data into sequences that can be fed into the model.

    :param data: List of PSG data arrays, each of shape [epoch_len, channels]
    :param labels: List of label arrays, each of shape [epoch_len]
    :param param: Dictionary of hyperparameters
    :param not_enhance: Whether to skip data enhancement
    :return: Tuple of preprocessed data and labels
    """
    big_group_size = param.get('big_group_size', 1000)
    sequence_epochs = param.get('sequence_epochs', 100)
    enhance_window_stride = param.get('enhance_window_stride', 50) if not_enhance else sequence_epochs

    # Convert lists to numpy arrays
    data = np.array(data)  # Shape: [num_samples, epoch_len, channels]
    labels = np.array(labels)  # Shape: [num_samples, epoch_len]

    # Normalize data
    data = normalization(data)

    # Divide data and labels into big groups
    data = data_big_group(data, big_group_size)  # [num_samples, num_big_groups, big_group_size, channels]
    labels = label_big_group(labels, big_group_size)  # [num_samples, num_big_groups, big_group_size]

    # Slice data and labels into windows
    data = data_window_slice(data, sequence_epochs,
                             enhance_window_stride)  # [num_samples, num_big_groups, num_windows, sequence_epochs, channels]
    labels = labels_window_slice(labels, sequence_epochs,
                                 enhance_window_stride)  # [num_samples, num_big_groups, num_windows, sequence_epochs]

    # Reshape data to combine samples, groups, and windows
    num_samples, num_big_groups, num_windows, seq_epochs, channels = data.shape
    data = data.reshape(num_samples * num_big_groups * num_windows, seq_epochs, channels)
    labels = labels.reshape(num_samples * num_big_groups * num_windows, seq_epochs)

    # Optionally, perform additional data enhancement here

    # One-hot encode labels if needed
    # Assuming labels are integers from 0 to num_classes - 1
    num_classes = param.get('num_classes', 5)
    labels = np.expand_dims(labels, axis=-1)  # Shape: [total_windows, sequence_epochs, 1]
    labels = np.repeat(labels, num_classes, axis=-1)  # Shape: [total_windows, sequence_epochs, num_classes]
    labels = labels.astype(np.float32)

    # Flatten the sequence_epochs dimension if your model expects single labels per window
    # For example, take the majority vote or the last label
    # Here, we'll take the last label for simplicity
    labels = labels[:, -1, :]  # Shape: [total_windows, num_classes]

    return data, labels


if __name__ == "__main__":
    from load_files import load_npz_files
    import yaml
    import glob
    import os

    # 读取超参数
    with open("config.yaml", encoding='utf-8') as f:
        hyper_params = yaml.full_load(f)

    # 加载所有 .npz 文件
    data_dir = r'.\sleepedf\prepared'
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    # 加载数据
    data, labels = load_npz_files(npz_files, hyper_params['input_channels'])

    # 预处理数据
    data, labels = preprocess(data, labels, hyper_params['preprocess'])

    print(f"Preprocessed data shape: {data.shape}")
    print(f"Preprocessed labels shape: {labels.shape}")

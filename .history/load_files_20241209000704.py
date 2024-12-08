import os
import glob
import numpy as np


def load_npz_file(npz_file: str) -> tuple:
    """
    加载单个npz文件
    
    :param npz_file: npz文件的完整路径
    :return: (数据, 标签, 采样率)的元组
    """
    print(f"Loading file: {npz_file}")
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


def load_npz_files(data_dir: str) -> list:
    """
    加载目录下的所有npz文件
    
    :param data_dir: 数据目录的路径
    :return: 包含所有数据的列表
    """
    # 获取所有npz文件的完整路径
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    
    print(f"Found {len(npz_files)} npz files:")
    for f in npz_files:
        print(f"  {f}")
    
    data_list = []
    labels_list = []
    fs = None

    for npz_f in npz_files:
        try:
            tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
            
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise ValueError(f"Sampling rate mismatch in {npz_f}")
            
            # 修改数据维度处理
            # 从 (1092, 3000, 3) 转换为 (1092, 3000, 3, 1, 1)
            tmp_data = tmp_data[..., np.newaxis, np.newaxis]
            
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data_list.append(tmp_data)
            labels_list.append(tmp_labels)
            
        except Exception as e:
            print(f"Error loading {npz_f}: {str(e)}")
            continue

    print(f"Successfully loaded {len(data_list)} files")
    return data_list, labels_list


if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    else:
        test_dir = "./sleepedf/prepared"
    
    try:
        data, labels = load_npz_files(test_dir)
        print(f"Test successful! Loaded {len(data)} files")
    except Exception as e:
        print(f"Test failed: {str(e)}")

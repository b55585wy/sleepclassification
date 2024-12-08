import os
import numpy as np
from typing import List, Tuple

def load_npz_files(data_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    加载npz文件中的数据和标签
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        数据和标签的列表，每个元素是(data, labels)元组
    """
    print(f"Loading data from {data_dir}")
    data_files = []
    
    # 遍历目录下的所有npz文件
    for file in sorted(os.listdir(data_dir)):
        if file.endswith('.npz'):
            file_path = os.path.join(data_dir, file)
            print(f"Loading file: {file_path}")
            
            try:
                # 直接加载数据和标签对
                loaded = np.load(file_path)
                data = loaded['data']  # 假设数据的key是'data'
                labels = loaded['labels']  # 假设标签的key是'labels'
                
                # 数据基本检查
                if len(data) != len(labels):
                    print(f"Warning: Data and labels length mismatch in {file}")
                    continue
                    
                # 保存数据和标签对
                data_files.append((data, labels))
                print(f"Loaded data shape: {data.shape}, labels shape: {labels.shape}")
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
    
    if not data_files:
        raise ValueError("No valid data files found!")
        
    print(f"Successfully loaded {len(data_files)} files")
    return data_files


if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    else:
        test_dir = "./sleepedf/prepared"
    
    try:
        data_files = load_npz_files(test_dir)
        print(f"Test successful! Loaded {len(data_files)} files")
    except Exception as e:
        print(f"Test failed: {str(e)}")

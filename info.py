import numpy as np

# 加载 npz 文件，允许加载 pickle 对象
data = np.load('data.npz', allow_pickle=True)

# 查看所有数组的名称
print("Arrays in the npz file:", data.files)

# 查看每个数组的形状和类型
for name in data.files:
    array = data[name]
    print(f"\nArray name: {name}")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    
    # 根据数据类型选择不同的显示方式
    if np.issubdtype(array.dtype, np.number):  # 数值类型
        print(f"First few elements: {array.flatten()[:5]}")
    elif array.dtype.kind == 'U':  # 字符串类型
        print(f"Elements: {array}")
    else:  # 其他类型（如对象）
        print("Object type data")

def inspect_npz(file_path):
    """
    详细检查 npz 文件的内容，包括样本统计信息
    """
    data = np.load(file_path, allow_pickle=True)
    print(f"=== Inspecting {file_path} ===\n")
    
    # 基本信息
    print(f"Number of arrays: {len(data.files)}")
    print(f"Array names: {data.files}\n")
    
    total_samples = 0  # 添加样本总数统计
    
    # 详细信息
    for name in data.files:
        array = data[name]
        print(f"Array: {name}")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")
        print(f"  Size in memory: {array.nbytes / 1024:.2f} KB")
        print(f"  Dimensions: {array.ndim}")
        
        # 计算样本数（通常是第一个维度）
        num_samples = array.shape[0] if array.shape else 1
        total_samples += num_samples
        print(f"  Number of samples: {num_samples}")
        
        # 根据数据类型选择不同的显示和统计方式
        if np.issubdtype(array.dtype, np.number):  # 数值类型
            print(f"  Min value: {array.min()}")
            print(f"  Max value: {array.max()}")
            print(f"  Mean value: {array.mean():.4f}")
            print(f"  Std deviation: {array.std():.4f}")
            print(f"  Sample data: {array.flatten()[:3]}...")
        elif array.dtype.kind == 'U':  # 字符串类型
            unique_values = np.unique(array)
            print(f"  Unique values: {len(unique_values)}")
            print(f"  Values: {array}")
        else:  # 对象类型
            print("  Object type data")
        print()
    
    print(f"=== Summary ===")
    print(f"Total number of samples across all arrays: {total_samples}")
    print(f"Total number of arrays: {len(data.files)}")

# 使用示例
inspect_npz('data.npz')


    
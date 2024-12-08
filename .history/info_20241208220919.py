import numpy as np

# 加载 npz 文件
data = np.load('data.npz')

# 查看所有数组的名称
print("Arrays in the npz file:", data.files)

# 查看每个数组的形状和类型
for name in data.files:
    array = data[name]
    print(f"\nArray name: {name}")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    print(f"First few elements: {array.flatten()[:5]}")  # 显示前5个元素

def inspect_npz(file_path):
    """
    详细检查 npz 文件的内容
    """
    data = np.load(file_path)
    print(f"=== Inspecting {file_path} ===\n")
    
    # 基本信息
    print(f"Number of arrays: {len(data.files)}")
    print(f"Array names: {data.files}\n")
    
    # 详细信息
    for name in data.files:
        array = data[name]
        print(f"Array: {name}")
        print(f"  Shape: {array.shape}")
        print(f"  Data type: {array.dtype}")
        print(f"  Size in memory: {array.nbytes / 1024:.2f} KB")
        print(f"  Dimensions: {array.ndim}")
        print(f"  Min value: {array.min()}")
        print(f"  Max value: {array.max()}")
        print(f"  Sample data: {array.flatten()[:3]}...")
        print()

# 使用示例
inspect_npz('path/to/your.npz')


    
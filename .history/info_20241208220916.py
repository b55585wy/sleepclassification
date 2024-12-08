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


    
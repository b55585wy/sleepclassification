import numpy as np
from tensorflow.keras.utils import Sequence

class SleepDataGenerator(Sequence):
    """睡眠数据生成器"""
    def __init__(self, data_list, labels_list, batch_size=32, sequence_length=20, shuffle=True):
        self.data_list = data_list
        self.labels_list = labels_list
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()
        
    def __len__(self):
        """每个epoch的批次数"""
        return int(np.ceil(len(self.data_list) / self.batch_size))
    
    def __getitem__(self, index):
        """获取一个批次的数据"""
        # 获取当前批次的文件索引
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.data_list))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # 处理这个批次的数据
        eeg_sequences = []
        eog_sequences = []
        label_sequences = []
        
        for idx in batch_indexes:
            # 获取数据和标签
            data = self.data_list[idx].astype(np.float32)
            labels = self.labels_list[idx]
            
            # 分离EEG和EOG数据
            eeg_data = data[..., :3]
            eog_data = data[..., -3:]
            
            # 创建序列
            for i in range(len(data) - self.sequence_length + 1):
                eeg_seq = eeg_data[i:i + self.sequence_length]
                eog_seq = eog_data[i:i + self.sequence_length]
                label_seq = labels[i:i + self.sequence_length]
                
                eeg_sequences.append(eeg_seq)
                eog_sequences.append(eog_seq)
                label_sequences.append(label_seq)
                
                if len(eeg_sequences) >= self.batch_size:
                    break
            
            if len(eeg_sequences) >= self.batch_size:
                break
        
        # 转换为numpy数组
        eeg_batch = np.array(eeg_sequences)
        eog_batch = np.array(eog_sequences)
        label_batch = np.array(label_sequences)
        
        # 转换标签为one-hot编码
        num_classes = 5  # 假设有5个类别
        label_batch_one_hot = np.eye(num_classes)[label_batch.astype(int)]
        
        return [eeg_batch, eog_batch], label_batch_one_hot
    
    def on_epoch_end(self):
        """每个epoch结束时调用"""
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle:
            np.random.shuffle(self.indexes) 
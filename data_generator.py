import numpy as np
from tensorflow.keras.utils import Sequence
from typing import List, Tuple

class SleepDataGenerator(Sequence):
    """睡眠数据生成器"""
    def __init__(self, data_files: List[Tuple[np.ndarray, np.ndarray]], 
                 batch_size=32, sequence_length=20, shuffle=True):
        self.data_files = data_files  # 现在是(data, labels)对的列表
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        
        # 预计算每个文件可以生成的序列数量
        self.file_sequences = []
        self.cumulative_sequences = [0]
        total_sequences = 0
        
        for data, _ in data_files:
            num_sequences = len(data) - sequence_length + 1
            self.file_sequences.append(num_sequences)
            total_sequences += num_sequences
            self.cumulative_sequences.append(total_sequences)
        
        self.total_sequences = total_sequences
        self.indexes = None
        self.on_epoch_end()
    
    def __len__(self):
        """每个epoch的批次数"""
        return int(np.ceil(len(self.data_files) / self.batch_size))
    
    def __getitem__(self, index):
        """获取一个批次的数据"""
        # 获取当前批次的文件索引
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.data_files))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # 处理这个批次的数据
        eeg_sequences = []
        eog_sequences = []
        label_sequences = []
        
        for idx in batch_indexes:
            # 找到序列所在的文件和起始位置
            file_idx, start_pos = self._find_file_index(idx)
            
            # 获取数据和标签
            data, labels = self.data_files[file_idx]
            data_seq = data[start_pos:start_pos + self.sequence_length]
            label_seq = labels[start_pos:start_pos + self.sequence_length]
            
            # 分离EEG和EOG数据
            eeg_data = data_seq[..., :3].astype(np.float32)
            eog_data = data_seq[..., -3:].astype(np.float32)
            
            eeg_sequences.append(eeg_data)
            eog_sequences.append(eog_data)
            label_sequences.append(label_seq)
            
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
        self.indexes = np.arange(len(self.data_files))
        if self.shuffle:
            np.random.shuffle(self.indexes) 
    
    def _find_file_index(self, seq_idx):
        """找到序列所在的文件和起始位置"""
        for file_idx, cum_seq in enumerate(self.cumulative_sequences):
            if seq_idx < cum_seq:
                start_pos = seq_idx - cum_seq + self.file_sequences[file_idx]
                return file_idx, start_pos
        raise ValueError("Sequence index out of range")
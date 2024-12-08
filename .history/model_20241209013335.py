import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from construct_model import create_u_encoder, create_mse, upsample 



def build_branch(input_tensor, filters, kernel_size, pooling_sizes, 
                dilation_sizes, activation, u_depths, u_inner_filter, 
                mse_filters, padding, pre_name):
    """构建单个分支
    
    Args:
        input_tensor: 输入张量 [batch, 20, 3000, 3, 1, 1]
        filters: 过滤器数量列表
        kernel_size: 卷积核大小（整数）
        pooling_sizes: 池化大小列表
        dilation_sizes: 扩张率列表
        activation: 激活函数
        u_depths: U型网络深度列表
        u_inner_filter: 内部过滤器数量
        mse_filters: MSE过滤器数量列表
        padding: 填充方式
        pre_name: 层名称前缀
    """
    l_name = f"{pre_name}_stream_enc"

    # encoder 1 
    # 确保pooling_size小于输入特征图大小
    safe_pooling_size = min(pooling_sizes[0], input_tensor.shape[2] // 4)
    u1 = create_u_encoder(input_tensor, filters[0], kernel_size, safe_pooling_size,
                         middle_layer_filter=u_inner_filter, depth=u_depths[0],
                         pre_name=l_name, idx=1, padding=padding, activation=activation)
    
    # 添加打印语句来调试
    print(f"U1 shape before reduce_dim: {u1.shape}")
    
    u1 = layers.Conv2D(int(filters[0] * 0.5), (1, 1),
                       name=f"{l_name}_reduce_dim_layer_1",
                       padding=padding, activation=activation)(u1)
                       
    print(f"U1 shape after reduce_dim: {u1.shape}")
    
    # 使用动态计算的池化大小
    current_size = u1.shape[1]
    safe_pool_size = min(pooling_sizes[0], current_size // 2)  # 确保池化后至少有一半大小
    pool = layers.MaxPooling2D((safe_pool_size, 1), 
                              strides=(safe_pool_size, 1),
                              name=f"{l_name}_pool1")(u1)
                              
    print(f"Pool1 shape: {pool.shape}")

    # encoder 2 
    safe_pooling_size = min(pooling_sizes[1], pool.shape[1] // 4)
    u2 = create_u_encoder(pool, filters[1], kernel_size, safe_pooling_size,
                         middle_layer_filter=u_inner_filter, depth=u_depths[1],
                         pre_name=l_name, idx=2, padding=padding, activation=activation)
    u2 = layers.Conv2D(int(filters[1] * 0.5), (1, 1),
                       name=f"{l_name}_reduce_dim_layer_2",
                       padding=padding, activation=activation)(u2)
    
    # 使用动态计算的池化大小
    current_size = u2.shape[1]
    safe_pool_size = min(pooling_sizes[1], current_size // 2)
    pool = layers.MaxPooling2D((safe_pool_size, 1), 
                              strides=(safe_pool_size, 1),
                              name=f"{l_name}_pool2")(u2)

    # MSE处理
    u1_mse = create_mse(u1, mse_filters[0], kernel_size=kernel_size,
                        dilation_rates=dilation_sizes, pre_name=l_name, idx=1,
                        padding=padding, activation=activation)
    u2_mse = create_mse(u2, mse_filters[1], kernel_size=kernel_size,
                        dilation_rates=dilation_sizes, pre_name=l_name, idx=2,
                        padding=padding, activation=activation)

    # 动态调整池化大小来下采样u1_mse
    current_size = u1_mse.shape[1]
    target_size = u2_mse.shape[1]
    pool_size = current_size // target_size
    u1_mse_down = layers.MaxPooling2D((pool_size, 1),
                                     strides=(pool_size, 1))(u1_mse)

    # 调整通道数以匹配u2_mse
    u1_mse_down = layers.Conv2D(mse_filters[1], (1, 1))(u1_mse_down)

    # 特征融合
    branch_merge = layers.concatenate([u1_mse_down, u2_mse], axis=-1,
                                    name=f"{l_name}_concat")
    
    # 通道调整
    branch_output = layers.Conv2D(filters[0], (1, 1),
                                 activation=activation,
                                 padding=padding,
                                 name=f"{l_name}_final")(branch_merge)

    return branch_output


def build_single_salient_model(params):
    """
    构建 SingleSalientModel 使用 Keras Functional API

    :param params: 超参数字典
    :return: Keras Model 实例
    """
    padding = params.get('padding', 'same')
    sleep_epoch_length = params['sleep_epoch_len']
    sequence_length = params['preprocess']['sequence_epochs']
    filters = params['train']['filters']
    kernel_size = params['train']['kernel_size']
    pooling_sizes = params['train']['pooling_sizes']
    dilation_sizes = params['train']['dilation_sizes']
    activation = params['train']['activation']
    u_depths = params['train']['u_depths']
    u_inner_filter = params['train']['u_inner_filter']
    mse_filters = params['train']['mse_filters']
    num_classes = params.get('num_classes', 5)

    # 定义输入层
    inputs = layers.Input(shape=(sequence_length * sleep_epoch_length, 1, 1), name='single_input')

    l_name = "single_model_enc"

    # Encoder 1
    u1 = create_u_encoder(inputs, filters[0], kernel_size, pooling_sizes[0],
                          middle_layer_filter=u_inner_filter, depth=u_depths[0],
                          pre_name=l_name, idx=1, padding=padding, activation=activation)
    u1 = layers.Conv2D(int(u1.shape[-1] * 0.5), (1, 1),
                       name=f"{l_name}_reduce_dim_layer_1",
                       padding=padding, activation=activation)(u1)
    pool1 = layers.MaxPooling2D((pooling_sizes[0], 1), name=f"{l_name}_pool1")(u1)

    # Encoder 2
    u2 = create_u_encoder(pool1, filters[1], kernel_size, pooling_sizes[1],
                          middle_layer_filter=u_inner_filter, depth=u_depths[1],
                          pre_name=l_name, idx=2, padding=padding, activation=activation)
    u2 = layers.Conv2D(int(u2.shape[-1] * 0.5), (1, 1),
                       name=f"{l_name}_reduce_dim_layer_2",
                       padding=padding, activation=activation)(u2)
    pool2 = layers.MaxPooling2D((pooling_sizes[1], 1), name=f"{l_name}_pool2")(u2)

    # Encoder 3
    u3 = create_u_encoder(pool2, filters[2], kernel_size, pooling_sizes[2],
                          middle_layer_filter=u_inner_filter, depth=u_depths[2],
                          pre_name=l_name, idx=3, padding=padding, activation=activation)
    u3 = layers.Conv2D(int(u3.shape[-1] * 0.5), (1, 1),
                       name=f"{l_name}_reduce_dim_layer_3",
                       padding=padding, activation=activation)(u3)
    pool3 = layers.MaxPooling2D((pooling_sizes[2], 1), name=f"{l_name}_pool3")(u3)

    # Encoder 4
    u4 = create_u_encoder(pool3, filters[3], kernel_size, pooling_sizes[3],
                          middle_layer_filter=u_inner_filter, depth=u_depths[3],
                          pre_name=l_name, idx=4, padding=padding, activation=activation)
    u4 = layers.Conv2D(int(u4.shape[-1] * 0.5), (1, 1),
                       name=f"{l_name}_reduce_dim_layer_4",
                       padding=padding, activation=activation)(u4)
    pool4 = layers.MaxPooling2D((pooling_sizes[3], 1), name=f"{l_name}_pool4")(u4)

    # Encoder 5
    u5 = create_u_encoder(pool4, filters[4], kernel_size, pooling_sizes[3],
                          middle_layer_filter=u_inner_filter, depth=u_depths[4],
                          pre_name=l_name, idx=5, padding=padding, activation=activation)
    u5 = layers.Conv2D(int(u5.shape[-1] * 0.5), (1, 1),
                       name=f"{l_name}_reduce_dim_layer_5",
                       padding=padding, activation=activation)(u5)

    # MSE Layers
    u1_mse = create_mse(u1, mse_filters[0], kernel_size=kernel_size,
                        dilation_rates=dilation_sizes, pre_name=l_name, idx=1,
                        padding=padding, activation=activation)
    u2_mse = create_mse(u2, mse_filters[1], kernel_size=kernel_size,
                        dilation_rates=dilation_sizes, pre_name=l_name, idx=2,
                        padding=padding, activation=activation)
    u3_mse = create_mse(u3, mse_filters[2], kernel_size=kernel_size,
                        dilation_rates=dilation_sizes, pre_name=l_name, idx=3,
                        padding=padding, activation=activation)
    u4_mse = create_mse(u4, mse_filters[3], kernel_size=kernel_size,
                        dilation_rates=dilation_sizes, pre_name=l_name, idx=4,
                        padding=padding, activation=activation)
    u5_mse = create_mse(u5, mse_filters[4], kernel_size=kernel_size,
                        dilation_rates=dilation_sizes, pre_name=l_name, idx=5,
                        padding=padding, activation=activation)

    # Decoder 4
    dec_l_name = "single_model_dec"
    up4 = upsample(u4_mse, pre_name=dec_l_name, idx=4)(u5_mse)
    concat4 = layers.concatenate([up4, u4_mse], axis=-1, name=f"{dec_l_name}_concat_4")
    d4 = create_u_encoder(concat4, filters[3], kernel_size, pooling_sizes[3],
                          middle_layer_filter=u_inner_filter, depth=u_depths[3],
                          pre_name=dec_l_name, idx=4, padding=padding, activation=activation)
    d4 = layers.Conv2D(int(d4.shape[-1] * 0.5), (1, 1),
                       name=f"{dec_l_name}_reduce_dim_4",
                       padding=padding, activation=activation)(d4)

    # Decoder 3
    up3 = upsample(u3_mse, pre_name=dec_l_name, idx=3)(d4)
    concat3 = layers.concatenate([up3, u3_mse], axis=-1, name=f"{dec_l_name}_concat_3")
    d3 = create_u_encoder(concat3, filters[2], kernel_size, pooling_sizes[2],
                          middle_layer_filter=u_inner_filter, depth=u_depths[2],
                          pre_name=dec_l_name, idx=3, padding=padding, activation=activation)
    d3 = layers.Conv2D(int(d3.shape[-1] * 0.5), (1, 1),
                       name=f"{dec_l_name}_reduce_dim_3",
                       padding=padding, activation=activation)(d3)

    # Decoder 2
    up2 = upsample(u2_mse, pre_name=dec_l_name, idx=2)(d3)
    concat2 = layers.concatenate([up2, u2_mse], axis=-1, name=f"{dec_l_name}_concat_2")
    d2 = create_u_encoder(concat2, filters[1], kernel_size, pooling_sizes[1],
                          middle_layer_filter=u_inner_filter, depth=u_depths[1],
                          pre_name=dec_l_name, idx=2, padding=padding, activation=activation)
    d2 = layers.Conv2D(int(d2.shape[-1] * 0.5), (1, 1),
                       name=f"{dec_l_name}_reduce_dim_2",
                       padding=padding, activation=activation)(d2)

    # Decoder 1
    up1 = upsample(u1_mse, pre_name=dec_l_name, idx=1)(d2)
    concat1 = layers.concatenate([up1, u1_mse], axis=-1, name=f"{dec_l_name}_concat_1")
    d1 = create_u_encoder(concat1, filters[0], kernel_size, pooling_sizes[0],
                          middle_layer_filter=u_inner_filter, depth=u_depths[0],
                          pre_name=dec_l_name, idx=1, padding=padding, activation=activation)

    # Zero Padding to match the desired sequence length
    pad_length = (sequence_length * sleep_epoch_length - d1.shape[1]) // 2
    zpad = layers.ZeroPadding2D(padding=((pad_length, pad_length), (0, 0)), name="zero_padding")(d1)

    # Reshape and final convolution
    reshape = layers.Reshape((sequence_length, sleep_epoch_length, filters[0]), name="reshape")(zpad)
    reshape = layers.Conv2D(filters[0], (1, 1), activation='tanh',
                            padding='same', name="final_conv")(reshape)
    pool = layers.AveragePooling2D((1, sleep_epoch_length), name="average_pool")(reshape)
    outputs = layers.Conv2D(num_classes, (kernel_size, 1), padding=padding,
                            activation='softmax', name="output_softmax")(pool)

    # 定义模型
    model = models.Model(inputs=inputs, outputs=outputs, name="SingleSalientModel")
    return model

def build_two_stream_salient_model(params):
    """构建双流显著性模型"""
    # 修改输入层形状以匹配数据
    eeg_input = layers.Input(shape=(20, 3000, 1),  # (sequence_length, time_points, channels)
                            name='EEG_input')
    eog_input = layers.Input(shape=(20, 3000, 1),
                            name='EOG_input')
    
    # 修改池化大小
    pooling_sizes = [2, 2, 2, 2]  # 使用更小的池化尺寸
    
    # EEG 流
    eeg_stream = build_branch(
        eeg_input, 
        filters=params['train']['filters'],
        kernel_size=params['train']['kernel_size'],
        pooling_sizes=pooling_sizes,  # 使用修改后的池化尺寸
        dilation_sizes=params['train']['dilation_sizes'],
        activation=params['train']['activation'],
        u_depths=params['train']['u_depths'],
        u_inner_filter=params['train']['u_inner_filter'],
        mse_filters=params['train']['mse_filters'],
        padding=params['train']['padding'],
        pre_name='EEG_stream'
    )
    
    # EOG 流
    eog_stream = build_branch(
        eog_input,
        filters=params['train']['filters'],
        kernel_size=params['train']['kernel_size'],
        pooling_sizes=pooling_sizes,  # 使用修改后的池化尺寸
        dilation_sizes=params['train']['dilation_sizes'],
        activation=params['train']['activation'],
        u_depths=params['train']['u_depths'],
        u_inner_filter=params['train']['u_inner_filter'],
        mse_filters=params['train']['mse_filters'],
        padding=params['train']['padding'],
        pre_name='EOG_stream'
    )
    
    # 合并两个流
    merged = layers.concatenate([eeg_stream, eog_stream])
    
    # 全连接层
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # 输出层 - 5分类
    outputs = layers.Dense(5, activation='softmax')(x)
    
    # 创建模型
    model = models.Model(inputs=[eeg_input, eog_input], outputs=outputs)
    
    return model

class SingleSalientModel(models.Model):
    """单流显著性模型"""
    def __init__(self, **params):
        super(SingleSalientModel, self).__init__()
        self.model = build_single_salient_model(params)

    def call(self, inputs):
        return self.model(inputs)

class TwoSteamSalientModelWrapper:
    """双流显著性模型包装器"""
    def __init__(self, params):
        self.model = build_two_stream_salient_model(params)

# 测试代码
if __name__ == '__main__':
    import yaml
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        hyp = yaml.full_load(f)
    model = TwoSteamSalientModelWrapper(hyp).model
    model.summary()

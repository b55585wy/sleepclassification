import tensorflow as tf
from tensorflow.keras import layers, models
from construct_model import create_u_encoder, create_mse, upsample


def build_branch(inputs, name_prefix, filters, kernel_size, pooling_sizes, 
                u_inner_filter, u_depths, padding, activation, 
                dilation_sizes, mse_filters):
    """
    构建双流模型中的单个分支（EEG或EOG）
    
    :param inputs: 输入张量
    :param name_prefix: 层名称前缀（EEG或EOG）
    :param filters: 卷积层滤波器数量列表
    :param kernel_size: 卷积核大小
    :param pooling_sizes: 池化大小列表
    :param u_inner_filter: U型网络内部滤波器数量
    :param u_depths: U型网络深度列表
    :param padding: 填充方式
    :param activation: 激活函数
    :param dilation_sizes: 空洞卷积率列表
    :param mse_filters: MSE层滤波器数量列表
    :return: 分支输出张量
    """
    l_name = f"{name_prefix}_branch"
    
    # Encoder 1
    u1 = create_u_encoder(inputs, filters[0], kernel_size, pooling_sizes[0],
                         middle_layer_filter=u_inner_filter, depth=u_depths[0],
                         pre_name=l_name, idx=1, padding=padding, activation=activation)
    u1 = layers.Conv2D(int(u1.shape[-1] * 0.5), (1, 1),
                       name=f"{l_name}_reduce_dim_1",
                       padding=padding, activation=activation)(u1)
    pool1 = layers.MaxPooling2D((pooling_sizes[0], 1), name=f"{l_name}_pool1")(u1)

    # MSE Layer
    u1_mse = create_mse(u1, mse_filters[0], kernel_size=kernel_size,
                        dilation_rates=dilation_sizes, pre_name=l_name, idx=1,
                        padding=padding, activation=activation)
    
    return u1_mse


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
    """
    构建双流显著性模型
    
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
    eeg_input = layers.Input(shape=(sequence_length * sleep_epoch_length, 1, 1), name='EEG_input')
    eog_input = layers.Input(shape=(sequence_length * sleep_epoch_length, 1, 1), name='EOG_input')

    # 构建 EEG 和 EOG 分支
    stream1 = build_branch(eeg_input, "EEG", filters, kernel_size, pooling_sizes, 
                          u_inner_filter, u_depths, padding, activation, 
                          dilation_sizes, mse_filters)
    stream2 = build_branch(eog_input, "EOG", filters, kernel_size, pooling_sizes,
                          u_inner_filter, u_depths, padding, activation,
                          dilation_sizes, mse_filters)

    # 特征融合
    mul = layers.multiply([stream1, stream2], name="multiply_streams")
    merge = layers.add([stream1, stream2, mul], name="add_streams")

    # 首先确保通道数正确
    merge = layers.Conv2D(filters[0], (1, 1), activation=activation,
                         padding=padding, name="channel_adjust")(merge)  # 调整到16通道
    
    # 计算正确的reshape维度
    total_time_steps = sequence_length * sleep_epoch_length  # 应该等于60000
    
    # 重要：确保维度匹配
    if merge.shape[1] != total_time_steps:
        raise ValueError(f"Expected time steps {total_time_steps}, but got {merge.shape[1]}")
    
    # 重塑维度
    reshape = layers.Reshape((sequence_length, sleep_epoch_length, filters[0]), 
                           name="reshape")(merge)
    
    # 输出层
    reshape = layers.Conv2D(filters[0], (1, 1), activation='tanh',
                           padding='same', name="final_conv")(reshape)
    pool = layers.AveragePooling2D((1, sleep_epoch_length), name="average_pool")(reshape)
    outputs = layers.Conv2D(num_classes, (kernel_size, 1), padding=padding,
                           activation='softmax', name="output_softmax")(pool)

    # 定义模型
    model = models.Model(inputs=[eeg_input, eog_input], outputs=outputs, name="TwoStreamSalientModel")
    return model


class SingleSalientModelWrapper:
    def __init__(self, params):
        self.model = build_single_salient_model(params)

    def summary(self):
        self.model.summary()


class TwoSteamSalientModelWrapper:
    def __init__(self, params):
        self.model = build_two_stream_salient_model(params)

    def summary(self):
        self.model.summary()


# 测试模型
if __name__ == '__main__':
    import yaml

    with open("config.yaml", encoding='utf-8') as f:
        hyp = yaml.safe_load(f)

    single_model = SingleSalientModelWrapper(hyp)
    single_model.summary()

    two_stream_model = TwoSteamSalientModelWrapper(hyp)
    two_stream_model.summary()

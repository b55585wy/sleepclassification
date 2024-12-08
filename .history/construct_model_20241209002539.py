from tensorflow.keras import layers
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


def upsample(dst: KerasTensor, pre_name: str = '', idx: int = 0) -> layers.Layer:
    """
    A customized up-sampling layer using bi-linear Interpolation
    :param dst: the target tensor, need it's size for up-sample
    :param pre_name: the layer's prefix name
    :param idx: the index of layer
    :return: the up-sample layer
    """
    from tensorflow import image
    import numpy as np
    np.random.rand(30)
    return layers.Lambda(lambda x, w, h: image.resize(x, (w, h)),
                         arguments={'w': dst.shape[1], 'h': dst.shape[2]}, name=f"{pre_name}_upsample{idx}")


def create_bn_conv(input: KerasTensor, filter: int, kernel_size: int, dilation_rate: int = 1, pre_name: str = '',
                   idx: int = 0, padding='same', activation: str = 'relu') -> KerasTensor:
    """
    A basic convolution-batchnormalization struct
    :param input: the input tensor
    :param filter: the filter number used in Conv layer
    :param kernel_size: the filter kernel size in Conv layer
    :param dilation_rate: the dilation rate for convolution
    :param pre_name: the layer's prefix name
    :param idx: the index of layer
    :param padding: same or valid
    :param activation: activation for Conv layer
    """
    conv = layers.Conv2D(filter, (kernel_size, 1), padding=padding,
                         dilation_rate=(dilation_rate, dilation_rate), activation=activation,
                         name=f"{pre_name}_conv{idx}")(input)
    bn = layers.BatchNormalization(name=f"{pre_name}_bn{idx}")(conv)
    return bn


def create_u_encoder(inputs, filters, kernel_size, pooling_size, middle_layer_filter,
                    depth, pre_name, idx, padding='same', activation='relu'):
    """创建U型编码器
    Args:
        inputs: 输入张量，形状为 [batch, 3000, 3]
               - batch: 批次大小
               - 3000: 时间步长（30秒 × 100Hz）
               - 3: 通道数（EEG, EOG, EMG）
    """
    l_name = f"{pre_name}_U{idx}_enc"
    
    # 验证池化大小参数
    if pooling_size <= 0:
        print(f"Warning: Invalid pooling_size {pooling_size} for encoder {idx}, using default value 2")
        pooling_size = 2
    
    print(f"Encoder {idx} - Input shape: {inputs.shape}, Pooling size: {pooling_size}")
    
    # 使用Conv1D处理时间序列数据，保持通道维度
    conv_bn = inputs
    for d in range(depth):
        # 一维卷积，在时间维度上进行
        conv_bn = layers.Conv1D(
            filters, kernel_size,
            padding=padding,
            activation=None,
            name=f"{l_name}_conv{d}"
        )(conv_bn)
        
        conv_bn = layers.BatchNormalization(
            name=f"{l_name}_bn{d}"
        )(conv_bn)
        
        conv_bn = layers.Activation(
            activation,
            name=f"{l_name}_act{d}"
        )(conv_bn)
        
        # 在时间维度上进行池化
        current_length = conv_bn.shape[1]  # 时间维度长度
        if idx < 5 and current_length > pooling_size:
            print(f"Encoder {idx}, Layer {d} - Shape before pooling: {conv_bn.shape}")
            safe_pool_size = min(pooling_size, current_length - 1)
            conv_bn = layers.MaxPooling1D(
                pool_size=safe_pool_size,
                strides=safe_pool_size,
                padding=padding,
                name=f"{l_name}_pool{d + 1}"
            )(conv_bn)
            print(f"Encoder {idx}, Layer {d} - Shape after pooling: {conv_bn.shape}")
        else:
            print(f"Encoder {idx}, Layer {d} - Skipping pooling, current shape: {conv_bn.shape}")
    
    # 中间层
    middle = layers.Conv1D(
        middle_layer_filter, kernel_size,
        padding=padding,
        activation=activation,
        name=f"{l_name}_middle"
    )(conv_bn)
    
    print(f"Encoder {idx} - Output shape: {middle.shape}")
    return middle

def create_mse(input: KerasTensor, filter: int, kernel_size: int, dilation_rates: list, pre_name: str = "",
               idx: int = 0, padding: str = 'same', activation: str = "relu") -> KerasTensor:
    """
    Multi-scale Extraction Module: a repetitive sub-structure of SalientSleepNet
    :param input: the input tensor
    :param filter: the filter number used in Conv layers
    :param kernel_size: the filter kernel size in Conv layers
    :param dilation_rates: the dilation rates for convolution
    :param pre_name: the layer's prefix name
    :param idx: the index of layer
    :param padding: same or valid
    :param activation: activation for Conv layer
    """
    l_name = f"{pre_name}_mse{idx}"

    convs = []
    for (i, dr) in enumerate(dilation_rates):
        conv_bn = create_bn_conv(input, filter, kernel_size, dilation_rate=dr,
                                 pre_name=l_name, idx=1 + i, padding=padding, activation=activation)
        convs.append(conv_bn)

    from functools import reduce
    con_conv = reduce(lambda l, r: layers.concatenate([l, r]), convs)

    down = layers.Conv2D(filter * 2, (kernel_size, 1), name=f"{l_name}_downconv1",
                         padding=padding, activation=activation)(con_conv)
    down = layers.Conv2D(filter, (kernel_size, 1), name=f"{l_name}_downconv2",
                         padding=padding, activation=activation)(down)
    out = layers.BatchNormalization(name=f"{l_name}_bn{len(dilation_rates) + 1}")(down)

    return out

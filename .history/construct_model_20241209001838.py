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
        inputs: 输入张量，形状为 [batch, 20, 3000, 3, 1, 1]
        filters: 卷积核数量
        kernel_size: 卷积核大小
        pooling_size: 池化大小
        middle_layer_filter: 中间层过滤器数量
        depth: 编码器深度
        pre_name: 层名称前缀
        idx: 编码器索引
    """
    l_name = f"{pre_name}_U{idx}_enc"
    
    # 验证池化大小参数
    if pooling_size <= 0:
        print(f"Warning: Invalid pooling_size {pooling_size} for encoder {idx}, using default value 2")
        pooling_size = 2
    
    print(f"Encoder {idx} - Input shape: {inputs.shape}, Pooling size: {pooling_size}")
    
    # 处理输入形状 [batch, 20, 3000, 3, 1, 1] -> [batch * 20, 3000, 3, 1]
    input_shape = inputs.shape
    batch_size = input_shape[0]  # None
    seq_len = input_shape[1]     # 20
    time_steps = input_shape[2]   # 3000
    channels = input_shape[3]     # 3
    
    # 重塑为 [batch * 20, 3000, 3, 1]
    reshaped = layers.Reshape(
        (batch_size * seq_len, time_steps, channels, 1),
        name=f"{l_name}_reshape"
    )(inputs)
    
    conv_bn = reshaped
    for d in range(depth):
        # 卷积层
        conv_bn = layers.Conv2D(
            filters, (kernel_size, 1),
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
        
        # 检查当前特征图的时间维度大小
        current_height = conv_bn.shape[1]  # 时间维度
        
        # 只在时间维度足够大时进行池化
        if idx < 5 and current_height > pooling_size:
            print(f"Encoder {idx}, Layer {d} - Shape before pooling: {conv_bn.shape}")
            safe_pool_size = min(pooling_size, current_height - 1)
            conv_bn = layers.MaxPooling2D(
                (safe_pool_size, 1),
                strides=(safe_pool_size, 1),
                padding=padding,
                name=f"{l_name}_pool{d + 1}"
            )(conv_bn)
            print(f"Encoder {idx}, Layer {d} - Shape after pooling: {conv_bn.shape}")
        else:
            print(f"Encoder {idx}, Layer {d} - Skipping pooling, current shape: {conv_bn.shape}")
    
    # 中间层
    middle = layers.Conv2D(
        middle_layer_filter, (kernel_size, 1),
        padding=padding,
        activation=activation,
        name=f"{l_name}_middle"
    )(conv_bn)
    
    # 重塑回序列形式 [batch * 20, new_time, channels, filters] -> [batch, 20, new_time, channels, filters]
    final_shape = middle.shape
    reshaped_middle = layers.Reshape(
        (batch_size, seq_len, final_shape[1], channels, final_shape[-1]),
        name=f"{l_name}_reshape_back"
    )(middle)
    
    print(f"Encoder {idx} - Output shape: {reshaped_middle.shape}")
    return reshaped_middle

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

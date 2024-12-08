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
        inputs: 输入张量，形状为 [batch, 3000, 3, 1] 或 [batch, 20, 3000, 3]
        filters: 卷积核数量
        kernel_size: 卷积核大小
        pooling_size: 池化大小
        middle_layer_filter: 中间层过滤器数量
        depth: 编码器深度
        pre_name: 层名称前缀
        idx: 编码器索引
        padding: 填充方式
        activation: 激活函数
    """
    l_name = f"{pre_name}_U{idx}_enc"
    
    # 验证池化大小参数
    if pooling_size <= 0:
        print(f"Warning: Invalid pooling_size {pooling_size} for encoder {idx}, using default value 2")
        pooling_size = 2
    
    print(f"Encoder {idx} - Input shape: {inputs.shape}, Pooling size: {pooling_size}")
    
    # 处理输入形状
    input_shape = inputs.shape
    if len(input_shape) == 4:  # [batch, 3000, 3, 1]
        conv_bn = inputs
    elif len(input_shape) == 5:  # [batch, 20, 3000, 3, 1]
        # 如果是序列输入，使用TimeDistributed包装器
        use_time_distributed = True
        conv_bn = inputs
    else:
        raise ValueError(f"Unexpected input shape: {input_shape}")
    
    # 编码器层
    for d in range(depth):
        # 卷积层
        if use_time_distributed:
            conv_bn = layers.TimeDistributed(
                layers.Conv2D(filters, (kernel_size, 1), 
                            padding=padding, 
                            activation=None),
                name=f"{l_name}_conv{d}"
            )(conv_bn)
            
            conv_bn = layers.TimeDistributed(
                layers.BatchNormalization(),
                name=f"{l_name}_bn{d}"
            )(conv_bn)
            
            conv_bn = layers.TimeDistributed(
                layers.Activation(activation),
                name=f"{l_name}_act{d}"
            )(conv_bn)
        else:
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
        time_dim_idx = 2 if use_time_distributed else 1
        current_height = conv_bn.shape[time_dim_idx]
        
        # 只在时间维度足够大时进行池化
        if idx < 5 and current_height > pooling_size:
            print(f"Encoder {idx}, Layer {d} - Shape before pooling: {conv_bn.shape}")
            safe_pool_size = min(pooling_size, current_height - 1)
            
            if use_time_distributed:
                conv_bn = layers.TimeDistributed(
                    layers.MaxPooling2D(
                        (safe_pool_size, 1),
                        strides=(safe_pool_size, 1),
                        padding=padding
                    ),
                    name=f"{l_name}_pool{d + 1}"
                )(conv_bn)
            else:
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
    if use_time_distributed:
        middle = layers.TimeDistributed(
            layers.Conv2D(
                middle_layer_filter, (kernel_size, 1),
                padding=padding,
                activation=activation
            ),
            name=f"{l_name}_middle"
        )(conv_bn)
    else:
        middle = layers.Conv2D(
            middle_layer_filter, (kernel_size, 1),
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

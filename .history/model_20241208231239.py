import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

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

    def call(self, inputs):
        return self.model(inputs)

class TwoSteamSalientModel(models.Model):
    def __init__(self, **params):
        super(TwoSteamSalientModel, self).__init__()
        self.params = params
        self.model = self._build_model()

    def _build_branch(self, inputs, name_prefix):
        """构建单个分支"""
        params = self.params
        u1 = create_u_encoder(inputs, params['train']['filters'][0],
                            params['train']['kernel_size'],
                            params['train']['pooling_sizes'][0],
                            params['train']['u_inner_filter'],
                            params['train']['u_depths'][0],
                            name_prefix, 1,
                            params.get('padding', 'same'),
                            params['train']['activation'])
        
        return create_mse(u1, params['train']['mse_filters'][0],
                         params['train']['kernel_size'],
                         params['train']['dilation_sizes'],
                         name_prefix, 1,
                         params.get('padding', 'same'),
                         params['train']['activation'])

    def _build_model(self):
        params = self.params
        padding = params.get('padding', 'same')
        sleep_epoch_length = params['sleep_epoch_len']
        sequence_length = params['preprocess']['sequence_epochs']
        filters = params['train']['filters']
        kernel_size = params['train']['kernel_size']
        num_classes = params.get('num_classes', 5)

        # 输入层
        eeg_input = layers.Input(shape=(sequence_length * sleep_epoch_length, 1, 1),
                               name='EEG_input')
        eog_input = layers.Input(shape=(sequence_length * sleep_epoch_length, 1, 1),
                               name='EOG_input')

        # 构建两个分支
        stream1 = self._build_branch(eeg_input, "EEG")
        stream2 = self._build_branch(eog_input, "EOG")

        # 特征融合
        mul = layers.multiply([stream1, stream2], name="multiply_streams")
        merge = layers.add([stream1, stream2, mul], name="add_streams")

        # 最终处理
        x = layers.Reshape((sequence_length, sleep_epoch_length, filters[0]),
                         name="reshape")(merge)
        x = layers.Conv2D(filters[0], (1, 1), activation='tanh',
                         padding='same', name="final_conv")(x)
        x = layers.AveragePooling2D((1, sleep_epoch_length),
                                  name="average_pool")(x)
        outputs = layers.Conv2D(num_classes, (kernel_size, 1),
                              padding=padding, activation='softmax',
                              name="output_softmax")(x)

        return models.Model(inputs=[eeg_input, eog_input],
                          outputs=outputs,
                          name="TwoStreamSalientModel")

    def call(self, inputs):
        return self.model(inputs)

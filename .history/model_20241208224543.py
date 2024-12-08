import tensorflow as tf
from tensorflow.keras import layers, models
from utils.construct_model import create_u_encoder, create_mse, upsample

class SingleSalientModel(models.Model):
    def __init__(self, **params):
        super(SingleSalientModel, self).__init__()
        self.params = params
        self.model = self._build_model()

    def _build_model(self):
        params = self.params
        padding = params.get('padding', 'same')
        sleep_epoch_length = params['sleep_epoch_len']
        sequence_length = params['preprocess']['sequence_epochs']
        filters = params['train']['filters']
        kernel_size = params['train']['kernel_size']
        pooling_sizes = params['train']['pooling_sizes']
        activation = params['train']['activation']
        u_depths = params['train']['u_depths']
        u_inner_filter = params['train']['u_inner_filter']
        mse_filters = params['train']['mse_filters']
        num_classes = params.get('num_classes', 5)

        inputs = layers.Input(shape=(sequence_length * sleep_epoch_length, 1, 1),
                            name='single_input')
        
        # Encoder 1
        u1 = create_u_encoder(inputs, filters[0], kernel_size, pooling_sizes[0],
                            u_inner_filter, u_depths[0], "single_model", 1,
                            padding, activation)
        u1_mse = create_mse(u1, mse_filters[0], kernel_size,
                           params['train']['dilation_sizes'],
                           "single_model", 1, padding, activation)

        # Final processing
        x = layers.Reshape((sequence_length, sleep_epoch_length, filters[0]),
                         name="reshape")(u1_mse)
        x = layers.Conv2D(filters[0], (1, 1), activation='tanh',
                         padding='same', name="final_conv")(x)
        x = layers.AveragePooling2D((1, sleep_epoch_length),
                                  name="average_pool")(x)
        outputs = layers.Conv2D(num_classes, (kernel_size, 1),
                              padding=padding, activation='softmax',
                              name="output_softmax")(x)

        return models.Model(inputs=inputs, outputs=outputs,
                          name="SingleSalientModel")

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

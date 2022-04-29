from __future__ import absolute_import,division,print_function

import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras import backend as K

WEIGHTS_PATH_X_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ 
    SepConv with BN between depthwise & pointwise. Optionally add activation after BN
    Implements right "same" padding for even kernel sizes
    Args:
        x: input tensor
        filters: num of filters in pointwise convolution
        prefix: prefix before name
        stride: stride at depthwise conv
        kernel_size: kernel size for depthwise convolution
        rate: atrous rate for depthwise convolution
        depth_activation: flag to use activation between depthwise & poinwise convs
        epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)

    return x

def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """
    Implements right 'same' padding for even kernel sizes
    Without this there is a 1 pixel drift when stride = 2
    Args:
        x: input tensor
        filters: num of filters in pointwise convolution
        prefix: prefix before name
        stride: stride at depthwise conv
        kernel_size: kernel size for depthwise convolution
        rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)

def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """
    Basic building block of modified Xception network
    Args:
        inputs: input tensor
        depth_list: number of filters in each SepConv layer. len(depth_list) == 3
        prefix: prefix before name
        skip_connection_type: one of {'conv','sum','none'}
        stride: stride at last depthwise conv
        rate: atrous rate for depthwise convolution
        depth_activation: flag to use activation between depthwise & pointwise convs
        return_skip: flag to return additional tensor after 2 SepConvs for decoder
    """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs

def DeepLabv3Plus(weights=None, input_tensor=None, input_shape=(None, None, 3), classes=2, backbone='xception', OS=16, activation=None):

    if not (weights in {'cityscapes', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cityscapes` ')

    if not (backbone in {'xception'}):
        raise ValueError('The `backbone` argument should be `xception` ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    """ 
    Start of Deeplabv3+ encoder
    Start of Xception (backbone/feature extractor)
    """
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    x = Conv2D(32, (3, 3), strides=(2, 2),
                name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation(tf.nn.relu)(x)
    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation(tf.nn.relu)(x)
    x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                skip_connection_type='conv', stride=2,
                                depth_activation=False, return_skip=True)
    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    for i in range(16):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)
    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)
    """ End of Xception """

    """ Atrous Spatial Pyramid Pooling (ASPP) Branching """
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(*shape_before[1:3], interpolation="bilinear")(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu, name='aspp0_activation')(b0)
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1', rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2', rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    """ Concatenate ASPP Branches """
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)

    """ Deeplabv3+ decoder """
    # skip_size = tf.keras.backend.int_shape(skip1)
    skip_size = tf.shape(x)*tf.constant((OS//4, OS//4, OS//4, OS//4))
    x = tf.keras.layers.experimental.preprocessing.Resizing(*skip_size[1:3], interpolation="bilinear")(x)
    dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)
    x = Conv2D(classes, (1, 1), padding='same', name='custom_logits_semantic')(x)
    size_before3 = tf.shape(img_input)
    x = tf.keras.layers.experimental.preprocessing.Resizing(*size_before3[1:3], interpolation="bilinear")(x)

    """ Optional activation layer """
    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)
    
    """ Model """
    model = Model(img_input, x, name='deeplabv3plus')

    """ Optional pre-trained weights """
    if weights == 'cityscapes':
        weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_X_CS,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    
    return model

if __name__ == '__main__':
    """ Just for testing purposes """
    model = DeepLabv3Plus(weights='cityscapes',input_shape=(None,None,3),classes=2,backbone='xception',activation='softmax')
    model.summary()

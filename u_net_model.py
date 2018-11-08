# @author : Abhishek R S

import os
import h5py
import numpy as np
import tensorflow as tf

'''
UNet
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition]
  (https://arxiv.org/abs/1409.1556)
- [UNet](https://arxiv.org/pdf/1505.04597)
- [UNet Project](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

# Pretrained model weights
- [Download pretrained vgg-16 model]
  (https://github.com/fchollet/deep-learning-models/releases/)
'''


class UNet:
    def __init__(self, pretrained_weights, is_training, data_format='channels_first', num_classes=15):
        self._weights_h5 = h5py.File(pretrained_weights, 'r')
        self._is_training = is_training
        self._data_format = data_format
        self._num_classes = num_classes
        self._padding = 'SAME'
        self._encoder_conv_strides = [1, 1, 1, 1]
        self._feature_map_axis = None
        self._encoder_data_format = None
        self._encoder_pool_kernel = None
        self._encoder_pool_strides = None
        self._initializer = tf.contrib.layers.xavier_initializer_conv2d()

        '''
        based on the data format set appropriate pool_kernel and pool_strides
        always use channels_first i.e. NCHW as the data format on a GPU
        '''

        if data_format == 'channels_first':
            self._encoder_data_format = 'NCHW'
            self._encoder_pool_kernel = [1, 1, 2, 2]
            self._encoder_pool_strides = [1, 1, 2, 2]
            self._feature_map_axis = 1
        else:
            self._encoder_data_format = 'NHWC'
            self._encoder_pool_kernel = [1, 2, 2, 1]
            self._encoder_pool_strides = [1, 2, 2, 1]
            self._feature_map_axis = -1

    # define vgg-16 encoder
    def vgg16_encoder(self, features):

        # input : BGR format with image_net mean subtracted
        # bgr mean : [103.939, 116.779, 123.68]

        if self._data_format == 'channels_last':
            features = tf.transpose(features, perm=[0, 2, 3, 1])

        # Stage 1
        self.conv1_1 = self._conv_block(features, 'block1_conv1')
        self.conv1_2 = self._conv_block(self.conv1_1, 'block1_conv2')
        self.pool1 = self._maxpool_layer(self.conv1_2, name='pool1')

        # Stage 2
        self.conv2_1 = self._conv_block(self.pool1, 'block2_conv1')
        self.conv2_2 = self._conv_block(self.conv2_1, 'block2_conv2')
        self.pool2 = self._maxpool_layer(self.conv2_2, name='pool2')

        # Stage 3
        self.conv3_1 = self._conv_block(self.pool2, 'block3_conv1')
        self.conv3_2 = self._conv_block(self.conv3_1, 'block3_conv2')
        self.conv3_3 = self._conv_block(self.conv3_2, 'block3_conv3')
        self.pool3 = self._maxpool_layer(self.conv3_3, name='pool3')

        # Stage 4
        self.conv4_1 = self._conv_block(self.pool3, 'block4_conv1')
        self.conv4_2 = self._conv_block(self.conv4_1, 'block4_conv2')
        self.conv4_3 = self._conv_block(self.conv4_2, 'block4_conv3')
        self.pool4 = self._maxpool_layer(self.conv4_3, name='pool4')

    # define u-net decoder
    def u_net_decoder(self):
        self.conv5 = self._get_conv2d_layer(
            self.pool4, 1024, [1, 1], [1, 1], name='conv5')
        self.bn5 = self._get_batchnorm_layer(self.conv5, name='bn5')
        self.elu5 = self._get_elu_activation(self.bn5, name='elu5')
        self.dropout5 = self._get_dropout_layer(self.elu5, name='dropout5')

        self.decoder1_out = self._get_decoder_block(
            self.dropout5, self.conv4_3, 512, name='decoder1_')
        self.decoder2_out = self._get_decoder_block(
            self.decoder1_out, self.conv3_3, 256, name='decoder2_')
        self.decoder3_out = self._get_decoder_block(
            self.decoder2_out, self.conv2_2, 128, name='decoder3_')
        self.decoder4_out = self._get_decoder_block(
            self.decoder3_out, self.conv1_2, 64, name='decoder4_')

        self.logits = self._get_conv2d_layer(self.decoder4_out, self._num_classes, [
                                             1, 1], [1, 1], name='logits')

    # return u-net decoder block
    def _get_decoder_block(self, features, features_to_concat, num_kernels, name='decoder_'):
        _conv_tr1 = self._get_conv2d_transpose_layer(
            features, num_kernels, [2, 2], [2, 2], name=name + 'conv_tr')
        _merged_features = tf.concat(
            [features_to_concat, _conv_tr1], axis=self._feature_map_axis, name=name + 'concat')

        _conv1 = self._get_conv2d_layer(_merged_features, num_kernels, [
                                        3, 3], [1, 1], name=name + 'conv1')
        _bn1 = self._get_batchnorm_layer(_conv1, name=name + 'bn1')
        _elu1 = self._get_elu_activation(_bn1, name=name + 'elu1')
        _dropout1 = self._get_dropout_layer(_elu1, name=name + 'dropout1')

        _conv2 = self._get_conv2d_layer(_dropout1, num_kernels, [3, 3], [
                                        1, 1], name=name + 'conv2')
        _bn2 = self._get_batchnorm_layer(_conv2, name=name + 'bn2')
        _elu2 = self._get_elu_activation(_bn2, name=name + 'elu2')
        _dropout2 = self._get_dropout_layer(_elu2, name=name + 'dropout2')

        return _dropout2

    # return convolution2d layer
    def _get_conv2d_layer(self, input_tensor, num_filters, kernel_size, strides, name='conv'):
        return tf.layers.conv2d(inputs=input_tensor, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name)

    # return convolution2d_transpose layer
    def _get_conv2d_transpose_layer(self, input_tensor, num_filters, kernel_size, strides, name='conv_tr'):
        return tf.layers.conv2d_transpose(inputs=input_tensor, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name)

    # return elu activation function
    def _get_elu_activation(self, input_tensor, name='elu'):
        return tf.nn.elu(input_tensor, name=name)

    # return dropout layer
    def _get_dropout_layer(self, input_tensor, rate=0.5, name='dropout'):
        return tf.layers.dropout(inputs=input_tensor, rate=rate, training=self._is_training, name=name)

    # return batch normalization layer
    def _get_batchnorm_layer(self, input_tensor, name='bn'):
        return tf.layers.batch_normalization(input_tensor, axis=self._feature_map_axis, training=self._is_training, name=name)

    #-------------------------------------#
    # pretrained vgg-16 encoder functions #
    #-------------------------------------#
    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_block(self, input_layer, name):
        W_init_value = np.array(
            self._weights_h5[name][name + '_W_1:0'], dtype=np.float32)
        b_init_value = np.array(
            self._weights_h5[name][name + '_b_1:0'], dtype=np.float32)

        W = tf.get_variable(name=name + '_kernel', shape=W_init_value.shape,
                            initializer=tf.constant_initializer(W_init_value), dtype=tf.float32)
        b = tf.get_variable(name=name + '_bias', shape=b_init_value.shape,
                            initializer=tf.constant_initializer(b_init_value), dtype=tf.float32)

        x = tf.nn.conv2d(input_layer, filter=W, strides=self._encoder_conv_strides,
                         padding=self._padding, data_format=self._encoder_data_format, name=name + '_conv')
        x = tf.nn.bias_add(
            x, b, data_format=self._encoder_data_format, name=name + '_bias')
        x = tf.nn.relu(x, name=name + '_relu')

        return x

    #-----------------------#
    # maxpool2d layer       #
    #-----------------------#
    def _maxpool_layer(self, input_layer, name):
        pool = tf.nn.max_pool(input_layer, ksize=self._encoder_pool_kernel, strides=self._encoder_pool_strides,
                              padding=self._padding, data_format=self._encoder_data_format, name=name)

        return pool

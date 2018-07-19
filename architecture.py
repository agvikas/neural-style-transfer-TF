from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os

class Model:

    def __init__(self, weights_path, trainable):
        self.weights_dict = np.load(weights_path, encoding='latin1').item()
        self.trainable = trainable
        print("VGG16 weight file loaded")

    def build(self, rand_img):

        self.conv1_1 = self._conv_layer(rand_img, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, "pool1")

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, "pool2")

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, "pool3")

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, "pool4")

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, "pool5")

    def _max_pool(self, feature_map, name):
        pool = tf.nn.max_pool(feature_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        return pool

    def _conv_layer(self, input, name):
        with tf.variable_scope(name) as scope:
            ker = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input, ker, [1, 1, 1, 1], padding='SAME')

            bias = self.get_bias(name)
            bias_add = tf.nn.bias_add(conv, bias)
            activation = tf.nn.relu(bias_add)
            return activation

    def get_conv_filter(self, name):
        initialize = tf.constant_initializer(value=self.weights_dict[name][0], dtype=tf.float32)
        shape = self.weights_dict[name][0].shape
        var = tf.get_variable(name="filter", initializer=initialize, shape=shape, trainable=self.trainable)
        return var

    def get_bias(self, name):
        initialize = tf.constant_initializer(value=self.weights_dict[name][1], dtype=tf.float32)
        shape = self.weights_dict[name][1].shape
        var = tf.get_variable(name="bias", initializer=initialize, shape=shape, trainable=self.trainable)
        return var

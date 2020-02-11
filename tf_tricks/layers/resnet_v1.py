"""
resnet_v1 blocks
https://arxiv.org/pdf/1512.03385.pdf

1 stage in ResNet: downsampling_block -> residual_block -> residual block
input is fed into a stem_block
"""
import tensorflow as tf

class StemBlock(tf.keras.layers.Layer):

    def __init__(self, output_filters=64, bn_momentum=.99, padding="same"):
        self.bn_momentum = bn_momentum
        self.output_filters = output_filters
        self.padding = padding
        super(StemBlock, self).__init__()

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(7, 7), strides=(2, 2), padding=self.padding)

        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

        self.activation_1 = tf.keras.layers.ReLU()

        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding=self.padding)

    def call(self, input):
        output = self.conv_1(input)
        output = self.bn_1(output)
        output = self.activation_1(output)
        output = self.max_pool(output)

        return output

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, output_filters, bn_momentum=.99, padding="same"):
        self.bn_momentum = bn_momentum
        self.output_filters = output_filters
        self.padding = padding
        super(ResidualBlock, self).__init__()

    def build(self, input_shape):

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.output_filters // 4, kernel_size=(1, 1), strides=(1, 1), padding=self.padding)
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.output_filters // 4, kernel_size=(3, 3), strides=(1, 1), padding=self.padding)
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(1, 1), strides=(1, 1), padding=self.padding)

        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.bn_3 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

        self.activation_1 = tf.keras.layers.ReLU()
        self.activation_2 = tf.keras.layers.ReLU()
        self.activation_3 = tf.keras.layers.ReLU()

    def call(self, input):
        skip = input
        output = self.conv_1(input)
        output = self.bn_1(output)
        output = self.activation_1(output)

        output = self.conv_2(output)
        output = self.bn_2(output)
        output = self.activation_2(output)

        output = self.conv_3(output)
        output = self.bn_3(output)

        output = output + skip
        output = self.activation_3(output)
        return output

class DownsamplingBlock(tf.keras.layers.Layer):

    def __init__(self, output_filters, bn_momentum=.99, padding="same"):
        self.bn_momentum = bn_momentum
        self.output_filters = output_filters
        self.padding = padding
        super(DownsamplingBlock, self).__init__()

    def build(self, input_shape):

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.output_filters // 4, kernel_size=(1, 1), strides=(2, 2), padding=self.padding)
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.output_filters // 4, kernel_size=(3, 3), strides=(1, 1), padding=self.padding)
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(1, 1), strides=(1, 1), padding=self.padding)

        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.bn_3 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

        self.activation_1 = tf.keras.layers.ReLU()
        self.activation_2 = tf.keras.layers.ReLU()
        self.activation_3 = tf.keras.layers.ReLU()

        self.skip_conv_1 = tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(1, 1), strides=(2, 2), padding=self.padding)
        self.skip_bn_1 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    def call(self, input):
        skip = self.skip_conv_1(input)
        skip = self.skip_bn_1(skip)

        output = self.conv_1(input)
        output = self.bn_1(output)
        output = self.activation_1(output)

        output = self.conv_2(output)
        output = self.bn_2(output)
        output = self.activation_2(output)

        output = self.conv_3(output)
        output = self.bn_3(output)

        output = output + skip
        output = self.activation_3(output)
        return output

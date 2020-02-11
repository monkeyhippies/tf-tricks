"""
resnet d blocks
from https://arxiv.org/abs/2001.06268

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
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(3, 3), strides=(2, 2), padding=self.padding)
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(3, 3), strides=(1, 1), padding=self.padding)
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(3, 3), strides=(1, 1), padding=self.padding)

        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.activation_1 = tf.keras.layers.ReLU()

        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.activation_2 = tf.keras.layers.ReLU()

        self.bn_3 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.activation_3 = tf.keras.layers.ReLU()

        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding=self.padding)

    def call(self, input):
        output = self.conv_1(input)
        output = self.bn_1(output)
        output = self.activation_1(output)

        output = self.conv_2(output)
        output = self.bn_2(output)
        output = self.activation_2(output)

        output = self.conv_3(output)
        output = self.bn_3(output)
        output = self.activation_3(output)


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

        self.skip_conv_1 =  tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(1, 1), strides=(1, 1), padding=self.padding)

        self.activation_1 = tf.keras.layers.ReLU()
        self.activation_2 = tf.keras.layers.ReLU()
        self.activation_3 = tf.keras.layers.ReLU()

    def call(self, input):
        output = self.bn_1(input)
        output = self.activation_1(output)

        skip = self.skip_conv_1(output)

        output = self.conv_1(output)

        output = self.bn_2(output)
        output = self.activation_2(output)
        output = self.conv_2(output)


        output = self.bn_3(output)
        output = self.activation_3(output)

        output = self.conv_3(output)

        output = output + skip
        return output

class DownsamplingBlock(tf.keras.layers.Layer):
    def __init__(self, output_filters, bn_momentum=.99, padding="same"):
        """
        NOTE: number of input filters must equal @output_filters
        """
        self.bn_momentum = bn_momentum
        self.output_filters = output_filters
        self.padding = padding
        super(DownsamplingBlock, self).__init__()

    def build(self, input_shape):

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.output_filters // 4, kernel_size=(1, 1), strides=(1, 1), padding=self.padding)
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.output_filters // 4, kernel_size=(3, 3), strides=(2, 2), padding=self.padding)
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(1, 1), strides=(1, 1), padding=self.padding)

        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.bn_3 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

        self.activation_1 = tf.keras.layers.ReLU()
        self.activation_2 = tf.keras.layers.ReLU()
        self.activation_3 = tf.keras.layers.ReLU()

        self.skip_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding=self.padding)
        self.skip_conv = tf.keras.layers.Conv2D(filters=self.output_filters, kernel_size=(1, 1), strides=(1, 1), padding=self.padding)

    def call(self, input):
        output = self.bn_1(input)
        output = self.activation_1(output)

        skip = self.skip_pooling(output)
        skip = self.skip_conv(skip)

        output = self.conv_1(output)

        output = self.bn_2(output)
        output = self.activation_2(output)
        output = self.conv_2(output)

        output = self.bn_3(output)
        output = self.activation_3(output)
        output = self.conv_3(output)

        output = output + skip
        return output

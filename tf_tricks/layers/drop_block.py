import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import tf_utils

from tf_tricks.nn.drop_block import drop_block

class DropBlock(tf.keras.layers.Layer):

    def __init__(self, block_size=(7, 7), rate=.1, seed=None):

        self.block_size = block_size
        self.rate = rate
        self.seed = seed

        super(DropBlock, self).__init__()

    def call(self, inputs, training=None):
        """
        Some code copied from https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/layers/core.py#L172
        """
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():

            return drop_block(
                inputs,
                noise_shape=array_ops.shape(inputs),
                block_size=self.block_size,
                rate=self.rate,
                seed=self.seed
            )

        output = tf_utils.smart_cond(
            training,
            dropped_inputs,
            lambda: array_ops.identity(inputs)
        )

        return output

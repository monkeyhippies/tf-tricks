import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops

def drop_block(x, noise_shape, block_size=(7, 7), rate=.1, seed=None):
    """
    Copied and editted from https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/ops/nn_ops.py#L4390
    @rate is percent chance of dropping out. It equals 1 - keep_prob
    channels are assumed to be last
    """
    rate = tf.cast(K.get_value(rate), x.dtype)

    # mask_shape is shape of x minus the padding added after M centers determined
    # x shape should be [batch_size, dim1, dim2, channels] 
    padding_size = 0, int(block_size[0] // 2), int(block_size[1] // 2), 0

    mask_shape = tuple(noise_shape[i] - (2 * padding_size[i]) for i in range(len(padding_size)))
    # Sample a uniform distribution on [0.0, 1.0) and select values larger than
    # rate.
    #
    # NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
    # and subtract 1.0.
    random_tensor = random_ops.random_uniform(
            mask_shape, seed=seed, dtype=x.dtype)
    keep_prob = 1.0 - rate
    noise_height = tf.cast(noise_shape[1], x.dtype)
    noise_width = tf.cast(noise_shape[2], x.dtype)
    block_height = tf.cast(block_size[0], x.dtype)
    block_width = tf.cast(block_size[1], x.dtype)

    gamma = (rate * noise_height * noise_width) / (
        block_height * block_width * \
        (noise_height - block_height + 1) * \
        (noise_width - block_width + 1)
    )
    # NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
    # float to be selected, hence we use a >= comparison.
    keep_mask = tf.cast(random_tensor >= gamma, x.dtype)
    keep_mask = tf.pad(keep_mask,
        paddings=[
            [padding_size[i], padding_size[i]]
            for i in range(len(padding_size)) 
        ], mode='CONSTANT', constant_values=1
    )
    # Dilate the M centers to block_size
    keep_mask = -(tf.nn.max_pool2d(-keep_mask, ksize=block_size, strides=1, padding="SAME"))
    scale_factor = tf.cast(noise_shape[1] * noise_shape[2], x.dtype) / (
        tf.reduce_sum(keep_mask, axis=[1, 2], keepdims=True) + 1e-8
    )
    ret = x * keep_mask * scale_factor
    ret = tf.cast(ret, x.dtype)

    return ret

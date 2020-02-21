"""
Big little network as described in https://arxiv.org/pdf/2001.06268.pdf
and https://arxiv.org/abs/1807.03848
"""

import tensorflow as tf

class BigLittleBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        block_cls,
        num_blocks,
        num_filters,
        downsampling_layer=None,
        filter_name="output_filters",
        alpha=2,
        beta=4,
        upsample_size=(2, 2),
        block_args=None,
        use_last_block=False,
    ):
        """
        @use_last_block: Boolean of whether adding final block
            after combining little and big
        @num_blocks: number of blocks in big network
        @alpha: ratio of big/little filters
        @beta: ratio of big/little blocks
        @block_args: arguments to pass to block_cls
        @downsampling_layer: instance of downsampling layer only used in big
        @upsample_size: rate of upsampling to apply to big to match little
            output (should equal to amount big downsamples by). If None, no
            upsampling is done
        """

        self.use_last_block = use_last_block
        self.block_cls = block_cls
        self.downsample = downsampling_layer
        self.block_args = block_args or {}

        self.num_filters = num_filters
        self.filter_name = filter_name
        self.alpha = alpha
        self.beta = beta
        self.upsample_size = upsample_size

        self.big_block_args = {
            k: v for k, v in self.block_args.items()
        }
        self.big_block_args[self.filter_name] = num_filters

        self.little_block_args = {
            k: v for k, v in self.block_args.items()
        }
        self.little_block_args[self.filter_name] = int(
            num_filters // self.alpha
        ) or 1

        self.num_big_blocks = num_blocks
        self.num_little_blocks = int(num_blocks // self.beta) or 1

        super(BigLittleBlock, self).__init__()

    def build(self, input_shape):

        self.big_layers = [
            self.block_cls(**self.big_block_args)
            for i in range(self.num_big_blocks)
        ]

        self.little_layers = [
            self.block_cls(**self.little_block_args)
            for i in range(self.num_little_blocks)
        ]

        # Final residual block as described in paper linked in class docstring
        # big is upsampled to size of little
        # conv layer increases little's filters to match big
        if self.use_last_block:
            self.residual_layer = self.block_cls(**self.big_block_args)
        else:
            self.residual_layer = None

        if self.upsample_size:
            self.upsample = tf.keras.layers.UpSampling2D(
                size=self.upsample_size, interpolation="bilinear"
            )
        else:
            self.upsample = None

        self.conv = tf.keras.layers.Conv2D(
            filters=self.num_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
        )

    def call(self, inputs):

        if self.downsample:
            big = self.downsample(inputs)
        else:
            big = inputs

        for layer in self.big_layers:
            big = layer(big)

        if self.upsample:
            big = self.upsample(big)

        little = inputs
        for layer in self.little_layers:
            little = layer(little)
        little = self.conv(little)

        if self.residual_layer:
            output = self.residual_layer(big + little)
        else:
            output = big + little

        return output

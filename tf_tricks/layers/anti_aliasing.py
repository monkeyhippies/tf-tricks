import tensorflow as tf
import numpy as np

def binomial_filter_1D(dim):
    """
    https://stackoverflow.com/questions/56246970/how-to-apply-a-binomial-low-pass-filter-to-data-in-a-numpy-array
    """

    power = dim - 1
    binomial_filter = np.poly1d([1, 1]) ** power
    binomial_filter = binomial_filter / (2 ** power)
    return binomial_filter.coeffs

def binomial_filter_2D(dims):

    filter_1 = np.reshape(
        binomial_filter_1D(dims[0]),
        (dims[0], 1)
    )
    filter_2 = np.reshape(
        binomial_filter_1D(dims[1]),
        (1, dims[1])
    )

    return np.matmul(filter_1, filter_2)

class BlurPool(tf.keras.layers.Layer):
    """
    blurpool using binomial filter as described in https://arxiv.org/pdf/1904.11486.pdf
    """
    def __init__(self, kernel_size=(3, 3), strides=(2, 2), padding="same"):

        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        self.untiled_kernel = binomial_filter_2D(
            self.kernel_size
        ).reshape(
            kernel_size + (1, )
        ).astype("float32")

        super(BlurPool, self).__init__()

    def build(self, input_shape):
        """
        Assume channels are last
        """

        self.num_filters = input_shape[-1]

        self.kernel = np.tile(self.untiled_kernel, (1, 1, self.num_filters))

        self.blur_pool = tf.keras.layers.DepthwiseConv2D(
            depthwise_initializer=tf.keras.initializers.Constant(
                self.kernel
            ),
            kernel_size=self.kernel_size,
            depth_multiplier=1,
            strides=self.strides,
            padding=self.padding,
            use_bias=False,
            activation=None,
            trainable=False
        )

    def call(self, input):

        return self.blur_pool(input)

class AAAveragePooling2D(BlurPool): pass

class AAStridedConv2D(tf.keras.layers.Layer):

    def __init__(self,
        blur_kernel_size=(3, 3), blur_strides=(2, 2), blur_padding="same",
        **kwargs
    ):
        """
        All of the conv variables should be written as keyword arguments
        """

        self.conv_kwargs = {
            "kernel_size": (3, 3),
            "strides": (2, 2),
            "padding": "same",
        }

        self.conv_kwargs.update(kwargs)

        self.blur_kernel_size = blur_kernel_size
        self.blur_strides = blur_strides
        self.blur_padding = blur_padding

        super(AAStridedConv2D, self).__init__()

    def build(self, input_shape):
        """
        Assume channels last
        """

        # Assume num filters same as input if not specified
        if "filters" not in self.conv_kwargs:
            self.conv_kwargs["filters"] = input_shape[-1]

        self.conv_1 = tf.keras.layers.Conv2D(
            **self.conv_kwargs
        )

        self.blur_pool = BlurPool(
            kernel_size=self.blur_kernel_size,
            strides=self.blur_strides,
            padding= self.blur_padding
        )

    def call(self, input):

        output = self.conv_1(input)
        output = self.blur_pool(output)

        return output

class AAMaxPooling2D(tf.keras.layers.Layer):

    def __init__(self,
        blur_kernel_size=(3, 3), blur_strides=(2, 2), blur_padding="same",
        pool_size=(3, 3), strides=(1, 1), padding="same"
    ):

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

        self.blur_kernel_size = blur_kernel_size
        self.blur_strides = blur_strides
        self.blur_padding = blur_padding

        super(AAMaxPooling2D, self).__init__()

    def build(self, input_shape):

        self.max_pooling = tf.keras.layers.MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding
        )

        self.blur_pool = BlurPool(
            kernel_size=self.blur_kernel_size,
            strides=self.blur_strides,
            padding= self.blur_padding
        )

    def call(self, input):

        output = self.max_pooling(input)
        output = self.blur_pool(output)

        return output

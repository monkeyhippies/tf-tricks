import tensorflow as tf

class SEBlock(tf.keras.layers.Layer):
    """
    Squeeze and excitation block per https://arxiv.org/pdf/1709.01507.pdf
    except without the initial convolution, so the input into SEBlock is assumed to be
    the output of some convolution
    """

    def __init__(self, r=2):
        """
        @r is dimensionality reduction ratio, as described in the paper SEBlock is based on
        """

        self.r = 2

        super(SEBlock, self).__init__()

    def build(self, input_shape):
        """
        channels dimension is assumed to be last
        """
        self.num_filters = input_shape[-1]
        self.hidden_units = int(self.num_filters // self.r)

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fully_connected = tf.keras.layers.Dense(self.hidden_units, activation="relu", use_bias=False)
        self.activations = tf.keras.layers.Dense(self.num_filters, activation="sigmoid", use_bias=False)
        # Note: attention shape should be (1, 1, ..., num_filters)
        self.activations_reshape = tf.keras.layers.Reshape([1] * (len(input_shape) - 2) + [self.num_filters])

    def call(self, input):

        scaling = self.global_pooling(input)
        scaling = self.flatten(scaling)
        scaling = self.fully_connected(scaling)
        scaling = self.activations(scaling)
        scaling = self.activations_reshape(scaling)

        return input * scaling

class SKConv(tf.keras.layers.Layer):
    """
    Selective Kernel Convolution as described in https://arxiv.org/pdf/2001.06268.pdf

    Note: Assumes channels are last dimension
    """

    def __init__(self, kernel_size_1=(3, 3), kernel_size_2=None, r=2, padding="same", bn_momentum=.99):
        """
        @r is reduction ratio for fuse step in the SKConv
        if @kernel_size_2 is None, then both kernels will be size @kernel_size_1
        and a single convolution is used instead of 2
        """
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2

        self.r = r
        self.padding = padding
        self.bn_momentum = bn_momentum

        super(SKConv, self).__init__()

    def build(self, input_shape):
        self.num_filters = input_shape[-1]
        self.z_units = int(self.num_filters // self.r)

        if self.kernel_size_2 is None:
            self.conv_1 = tf.keras.layers.Conv2D(
                filters=self.num_filters * 2,
                kernel_size=self.kernel_size_1,
                strides=(1, 1), padding=self.padding
            )
            self.conv_2 = None
        else:            
            self.conv_1 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=self.kernel_size_1,
                strides=(1, 1),
                padding=self.padding
            )
            self.conv_2 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=self.kernel_size_2,
                strides=(1, 1),
                padding=self.padding
            )

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fully_connected = tf.keras.layers.Dense(self.z_units, activation=None, use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
        self.activation = tf.keras.layers.ReLU()
        self.soft_attention = tf.keras.layers.Dense(self.num_filters, activation="softmax", use_bias=False)
        # Note: attention shape should be (1, 1, ..., num_filters)
        self.attention_reshape = tf.keras.layers.Reshape([1] * (len(input_shape) - 2) + [self.num_filters])

    def call(self, input):

        if self.kernel_size_2 is None:
            output = self.conv_1(input)
            output_1, output_2 = tf.split(output, num_or_size_splits=2, axis=-1)
        else:
            output_1 = self.conv_1(input)
            output_2 = self.conv_2(input)

        attention = output_1 + output_2
        attention = self.global_pooling(attention)
        attention = self.flatten(attention)
        attention = self.fully_connected(attention)
        attention = self.batch_norm(attention)
        attention = self.activation(attention)

        # attention_1 + attention_2 = 1
        attention_1 = self.soft_attention(attention)
        attention_1 = self.attention_reshape(attention_1)
        attention_2 = 1.0 - attention_1

        output = output_1 * attention_1 + output_2 * attention_2

        return output

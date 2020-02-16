import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

class LinearDecay(Callback):
    """
    Callback for linearly decaying a tensor variable

    Example:
    ```
    linear_decay = LinearDecay(value=None, start_value=1.0, end_value=.9, steps=1e6)
    model.fit(x_train, y_train, callbacks=[linear_decay])
    linear_decay.value
    ```
    """

    def __init__(self,
        value=None,
        start_value=None,
        end_value=.9,
        steps=1e6,
        global_step=None
    ):
        """
        @value is a tf.Variable whose value this callback updates
            if @value is None, a new tf.Variable object is set to self.value
        @start_value is the start value of self.value
        @end_value is the end value of self.value
        @steps is number of steps to linearly decay self.value
        @global_step is tf.Variable with global training step value
        """

        super(LinearDecay, self).__init__()

        if value is None:
            if start_value is None:
                raise Exception(
                    "If value isn't set, start_value must be specified"
                )

            else:
                self.value = tf.Variable(start_value, dtype=tf.float32)

        else:
            self.value = value
            if start_value is not None:
                K.set_value(self.value, start_value)

        self.start_value = float(K.get_value(self.value))
        self.end_value = float(end_value)
        self.steps = int(steps)
        self.global_step = global_step or tf.Variable(0, dtype=tf.int32)

   
    def on_batch_begin(self, batch, logs={}):

        step = tf.minimum(self.global_step, self.steps)
        progress = tf.cast(step, tf.float32) / tf.cast(self.steps, tf.float32)

        new_value = (self.end_value - self.start_value) * progress + \
            self.start_value

        self.value.assign(new_value)

    def on_batch_end(self, batch, logs={}):
        self.global_step.assign_add(1)

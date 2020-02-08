import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

class CyclicLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    https://arxiv.org/abs/1506.01186
    """

    def __init__(self, stepsize, max_lr, min_lr=None, triangular2=True):
        """
        @stepsize is half a cycle
        stepsize is recommended in range [2, 8] * (steps_per_epoch)
        """

        self.stepsize = stepsize
        self.max_lr = max_lr
        self.min_lr = max_lr / 10 if min_lr is None else min_lr
        self.triangular2 = triangular2

    def get_config(self):

        return {
                stepsize: self.stepsize,
                min_lr: self.min_lr,
                max_lr: self.max_lr,
                triangular2: self.triangular2
        }

    def __call__(self, step):
        max_lr = ops.convert_to_tensor(self.max_lr, name="max_learning_rate")
        dtype = max_lr.dtype
        stepsize = math_ops.cast(self.stepsize, dtype)
        min_lr = math_ops.cast(self.min_lr, dtype)

        cycle = math_ops.floor(math_ops.divide(step, stepsize * 2))
        if self.triangular2:
            max_lr = (max_lr - min_lr) / (2 ** cycle) + min_lr
        # relative is in [-1, 1], representing what phase in cycle it is, starting at -1 and ending at 1
        relative = (step - cycle * stepsize * 2 - stepsize) / stepsize
        lr = math_ops.abs(relative) * (min_lr - max_lr) + max_lr
        lr = math_ops.maximum(math_ops.cast(0, dtype), lr)
        return lr

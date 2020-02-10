from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule

def with_lr_multipliers(optimizer_class):
    """
    @optimizer_class is an optimizer subclassed from TF2.0's optimizer_v2
    outputs a subclassed optimizer with lr_multiplier capabilities

    Example:
    Below is code to create an Adam optimizer object that has a quarter the learning rate
    for layer_1 and half the learning rate for layer_2
    ```
    import tensorflow.keras.optimizers.Adam as Adam

    LRMultiplierAdam = with_lr_multipliers(Adam)
    lr_multipliers = {
        "layer_1": .25,
        "layer_2": .5
    }
    optimizer = LRMultiplierAdam(learning_rate=.04, lr_multipliers=lr_multipliers)
    ```
    """

    class LRMultiplierOptimizer(optimizer_class):

        def __init__(self, lr_multipliers=None, *args, **kwargs):
            """
            @lr_multipliers is a dict mapping variable name to lr multiplier
            """
            self.lr_multipliers = lr_multipliers or {}
            super(LRMultiplierOptimizer, self).__init__(*args, **kwargs)
        
        def _resource_apply_dense(self, grad, var, apply_state):
            var_name = var.name.split("/")[0]
            multiplier = self.lr_multipliers.get(var_name)
            if multiplier is not None:
                orig_learning_rate = self._hyper["learning_rate"]
                if isinstance(orig_learning_rate, learning_rate_schedule.LearningRateSchedule):
                    var_dtype = var.dtype
                    step = math_ops.cast(self.iterations, var_dtype)
                    learning_rate = math_ops.cast(orig_learning_rate(step), var_dtype)
                else:
                    learning_rate = K.get_value(orig_learning_rate)
                self._set_hyper("learning_rate", learning_rate * float(multiplier))
                # Dont use apply_state because it's impossible to do generic multipliers wrapper
                # with it since there's no standard for the learning_rate keyword in apply_state
                output = super(LRMultiplierOptimizer, self)._resource_apply_dense(grad, var, None)
                self._hyper["learning_rate"] = orig_learning_rate
            else:
                output = super(LRMultiplierOptimizer, self)._resource_apply_dense(grad, var, apply_state)

            return output

        def _resource_apply_sparse(self, grad, var, indices, apply_state):
            var_name = var.name.split("/")[0]
            multiplier = self.lr_multipliers.get(var_name)
            if multiplier is not None:
                orig_learning_rate = self._hyper["learning_rate"]
                if isinstance(orig_learning_rate, learning_rate_schedule.LearningRateSchedule):
                    var_dtype = var.dtype
                    step = math_ops.cast(self.iterations, var_dtype)
                    learning_rate = math_ops.cast(orig_learning_rate(step), var_dtype)
                else:
                    learning_rate = K.get_value(orig_learning_rate)
                self._set_hyper("learning_rate", learning_rate * float(multiplier))
                # Dont use apply_state because it's impossible to do generic multipliers wrapper
                # with it since there's no standard for the learning_rate keyword in apply_state
                output = super(LRMultiplierOptimizer, self)._resource_apply_sparse(grad, var, indices, None)
                self._hyper["learning_rate"] = orig_learning_rate
            else:
                output = super(LRMultiplierOptimizer, self)._resource_apply_sparse(grad, var, indices, apply_state)
    
            return output

        def get_config(self):
            config =    super(LRMultiplierOptimizer, self).get_config()
            config["lr_multipliers"] = self.lr_multipliers

            return config

    return LRMultiplierOptimizer

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

class LRFinder(Callback):
    """
    Callback for finding ideal learning rate, inspired by https://docs.fast.ai/callbacks.lr_finder.html

    Example:
    ```
    lr_finder = LRFinder(end_lr=.01)
    model.fit(x_train, y_train, callbacks=[lr_finder])
    learning_rates = lr_finder.lrs
    losses = lr_finder.losses
    ```
    """

    def __init__(self, start_lr=1e-7, end_lr=1e-1, num_iterations=100):
        super(LRFinder, self).__init__()
        self.start_lr = float(start_lr)
        self.end_lr = float(end_lr)
        self.num_iterations = num_iterations

    def on_train_begin(self, logs=None):
        self.lrs = []
        self.losses = []

    def get_lr(self, batch):
        # scale lr by multiplicative factor for each batch
        lr = self.start_lr * ((self.end_lr / self.start_lr) ** (float(batch) / self.num_iterations))
        return K.get_value(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = self.get_lr(batch)
        K.set_value(self.model.optimizer.lr, lr)
    
    def on_batch_end(self, batch, logs={}):
        self.lrs.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs.get("loss"))
        if batch >= self.num_iterations - 1:
            self.model.stop_training = True

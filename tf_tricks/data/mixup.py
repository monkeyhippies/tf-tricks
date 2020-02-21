"""
Utils for mixup as described in https://arxiv.org/pdf/1710.09412.pdf
"""
import numpy as np

def mixup_generator(gen_1, gen_2, alpha=.4):
    """
    outputs a generator with mixup performed on image data
    @alpha is hyperparamter to output lambda
    @gen_1: data generator that outputs tuple
        (image<np_array>, one_hot_encoded_label<np_array>)
    @gen_2: like @gen1
    """

    lam = np.random.beta(alpha, alpha)
    for (image_1, label_1), (image_2, label_2) in zip(gen_1, gen_2):
        image = lam * image_1 + (1 - lam) * image_2
        label = lam * label_1 + (1 - lam) * label_2

        yield image, label

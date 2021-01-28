import sys, os
import tensorflow as tf
import numpy as np
import scipy.stats as stats
import scipy.io
from PIL import Image

from tensorflow.keras.losses import Loss


class CRAFTLoss(Loss):
    def __init__(self, ):
        super(CRAFTLoss, self).__init__()
        
    def call(self, y_target_maps, y_pred_maps, mode='SynthData'):
        '''
        y_target_maps: numpy array of shape (B x H x W x 2)
        y_pred_maps: numpy array of size (B x H x W x 2)
        '''
        L = 0
        if mode == 'SynthData':
            L = tf.nn.l2_loss(y_pred_maps - y_target_maps)
        else:
            # here will be implemented the modified loss function for training on non-synthetic data
            pass

        return L


if __name__ == '__main__':
    craft_loss = CRAFTLoss()
    sample_y1 = np.random.rand(10, 400, 500, 2)
    sample_y2 = np.random.rand(10, 400, 500, 2)

    print(f'sample loss: {craft_loss(sample_y1, sample_y2)}')
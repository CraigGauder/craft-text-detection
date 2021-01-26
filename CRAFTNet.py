import sys, os
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, ReLU, UpSampling2D
from tensorflow.keras import Sequential


class DownConvBlock(tf.Module):
    def __init__(self, out_channels, pooling=True, name=None):
        super(DownConvBlock, self).__init__()
        self.conv_layers = Sequential(layers=(
            Conv2D(out_channels, 3, padding='same'),
            BatchNormalization(),
            ReLU()
        ))
        if pooling:
            self.pooling_layer = MaxPool2D()
        else:
            self.pooling_layer = lambda x: x
    def __call__(self, x):
        x = self.conv_layers(x)
        x = self.pooling_layer(x)
        return x

class UpConvBlock(tf.Module):
    def __init__(self, out_channels):
        super(UpConvBlock, self).__init__()
        self.conv_layers = Sequential(layers=(
            Conv2D(out_channels*2, 1, padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(out_channels, 3, padding='same'),
            BatchNormalization(),
            ReLU()
        ))
    def __call__(self, x):
        x = self.conv_layers(x)
        return x
    
class CRAFTNet(tf.Module):
    def __init__(self):
        super(CRAFTNet, self).__init__()
    
        self.downconv1 = DownConvBlock(64)
        self.downconv2 = DownConvBlock(128)
        self.downconv3 = DownConvBlock(256)
        self.downconv4 = DownConvBlock(512)
        self.downconv5 = DownConvBlock(512)
        self.downconv6 = DownConvBlock(512, pooling=False)
        
        self.upconv1 = UpConvBlock(256)
        self.upconv2 = UpConvBlock(128)
        self.upconv3 = UpConvBlock(64)
        self.upconv4 = UpConvBlock(32)
        
        self.finalConv = Sequential(layers=(
            Conv2D(32, 3, padding='same'), ReLU(),
            Conv2D(32, 3, padding='same'), ReLU(),
            Conv2D(16, 3, padding='same'), ReLU(),
            Conv2D(16, 1, padding='same'), ReLU(),
            Conv2D(2, 1, padding='same'),
        ))
        
        self.lastUpsample = UpSampling2D(interpolation='bilinear')
        
    def __call__(self, x):
        #VGG16-BN
        vgg16 = []
        z = self.downconv1(x)
        vgg16.append(z)
        z = self.downconv2(z)
        vgg16.append(z)
        z = self.downconv3(z)
        vgg16.append(z)
        z = self.downconv4(z)
        vgg16.append(z)
        z = self.downconv5(z)
        vgg16.append(z)
        z = self.downconv6(z)
        vgg16.append(z)
        
        #UpConv Network
        z = self.upconv1(tf.concat([z, vgg16[-2]], 3))
        z = tf.image.resize(z, vgg16[-3].shape[1:3])
        
        z = self.upconv2(tf.concat([z, vgg16[-3]], 3))
        z = tf.image.resize(z, vgg16[-4].shape[1:3])
        
        z = self.upconv3(tf.concat([z, vgg16[-4]], 3))
        z = tf.image.resize(z, vgg16[-5].shape[1:3])
        
        z = self.upconv4(tf.concat([z, vgg16[-5]], 3))
        feature = z
        
        #Final Conv to Output
        z = tf.image.resize(z, vgg16[-6].shape[1:3])
        y = self.finalConv(z)
        
        return y, feature


if __name__ == '__main__':
    craft_net = CRAFTNet()

    x = tf.convert_to_tensor(np.random.randint(255, size=(5, 400, 600, 3)).astype(np.float32), dtype="float")
    out, feat = craft_net(x)
    print('in shape:', x.shape)
    print('out shape:', out.shape)

    assert((np.array(x.shape[1:3]) == np.array(out.shape[1:3])*2).all())


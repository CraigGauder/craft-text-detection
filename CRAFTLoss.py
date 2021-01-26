import sys, os
import tensorflow as tf
import numpy as np
import scipy.stats as stats
import scipy.io
from PIL import Image

import h5py
import pickle
import cv2
import matplotlib.pyplot as plt
from shapely import geometry as geo

import torch
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ReLU, Module, Sequential
from torch.autograd import Variable
from torch.optim import adam


class CRAFTLoss(Module):
    def __init__(self):
        super(CRAFTLoss, self).__init__()
        
    def forward(self, y_pred_maps, y_target_maps):
        '''
        y_pred_maps: numpy array of size (B x 2 x H x W)
        y_target_maps: numpy array of shape (B x 2 x H x W)
        '''
        pass
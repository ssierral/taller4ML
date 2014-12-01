# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 15:30:47 2014

@author: elvert
"""
from scipy import io
import numpy as np


def sampleIMAGES():
    IMAGES = io.loadmat('IMAGES.mat')
    arrayIMAGES=np.array(IMAGES.get('IMAGES'))
    patchsize = 8
    numpatches = 10000
    patches=np.zeros((patchsize*patchsize, numpatches))
    images_idx = np.zeros((1, numpatches))
    for i in range(numpatches):
        chosen_image_idx=randrange(1,10)
        images_idx[:,i]=chosen_image_idx
        row=randrange(512-patchsize)
        col=randrange(512-patchsize)
        patches[:, i] = arrayIMAGES[row:row+patchsize, col:col+patchsize, chosen_image_idx].reshape((1, 64))
    return patches

if __name__ == '__main__':
    patches= sampleIMAGES()
    print len(patches)
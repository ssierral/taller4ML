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

def sigmoid(x):  
    return 1 / (1 + np.exp(-x));
    
def sparseAutoencoderCost(theta, visibleSize, hiddenSize,lambdaa, sparsityParam, beta, data):
    W1 = np.reshape(theta[1:hiddenSize*visibleSize], (hiddenSize, visibleSize))
    W2 = np.reshape(theta[hiddenSize*visibleSize+1:2*hiddenSize*visibleSize],(visibleSize,hiddenSize))
    b1 = theta[2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize]
    b2 = theta[2*hiddenSize*visibleSize+hiddenSize+1:len(theta)]
    cost = 0;
    W1grad = np.zeros(len(W1)) 
    W2grad = np.zeros(len(W2))
    b1grad = np.zeros(len(b1)) 
    b2grad = np.zeros(len(b2)) 

    
    grad = np.array([W1grad[:],W2grad[:],b1grad[:],b2grad[:]])    
    yield cost
    yield grad    

if __name__ == '__main__':
    patches= sampleIMAGES()
    print len(patches)
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 14:31:24 2014

@author: elvert
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:36:27 2014

@author: elvert
"""
import numpy as np
from pylab import *
import os, struct
from array import array
from cvxopt.base import matrix
from scipy.optimize import minimize
from scipy import sparse
from scipy import misc
import glob
       
def read(digits, dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    images =  matrix(0, (len(ind), rows*cols))
    labels = matrix(0, (len(ind), 1))
    for i in xrange(len(ind)):
        images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        labels[i] = lbl[ind[i]]

    return images, labels
#STEP 2: Implement softmaxCost  
def softmaxCost(theta, numClasses, inputSize, lambdaa, inputData, labels):
    theta = theta.reshape(numClasses, inputSize)
    numCases = inputData.shape[1]
    #groundTruth = full(sparse(labels, 1:numCases, 1));     
    groundTruth = np.array(sparse.csr_matrix((np.ones(numCases), (labels, np.array(range(numCases))))).todense())
    cost = 0
    ## ---------- YOUR CODE HERE --------------------------------------
    tmp = theta.reshape(numClasses, inputSize).dot(inputData)
    tmp = tmp-np.max(tmp)
    prob = np.exp(tmp) / np.sum(np.exp(tmp), axis=0)
    cost = (-1 / numCases) * np.sum(groundTruth * np.log(prob)) + (lambdaa/2) * np.sum(theta * theta)
    grad = (-1 / numCases) * (groundTruth - prob).dot(inputData.T) + lambdaa * theta
    return cost, grad.flatten()
    
def softmaxTrain(inputSize, numClasses, lambdaa, inputData, labels, options): 
    theta = 0.005 * np.random.randn(numClasses * inputSize)
    J = lambda x: softmaxCost(x, numClasses, inputSize, lambdaa, inputData, labels)

    result = minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
    return result.x.reshape(numClasses, inputSize)
    
def softmaxPredict(optTheta, inputData):
    prod = optTheta.dot(inputData)
    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    return pred.argmax(axis=0)

def importFaces():
    
    path = './jaffe/*.tiff'   
    files = glob.glob(path)
    labelsTr=[]
    imagesTr=[]
    labelsTs=[]
    imagesTs=[]   
    for name in files:         
        tmpImg = misc.imread(name)
        tmpImg = misc.imresize(tmpImg,(64,64),'nearest')
        tmpImg = tmpImg.flatten()
        tmp = name.split('.')
        lab = 0
        if tmp[2][0]+tmp[2][1]=='NE':
            lab = 1
        elif tmp[2][0]+tmp[2][1]=='HA':
            lab = 2
        elif tmp[2][0]+tmp[2][1]=='SA':
            lab = 3
        elif tmp[2][0]+tmp[2][1]=='SU':
            lab = 4
        elif tmp[2][0]+tmp[2][1]=='AN':
            lab = 5
        elif tmp[2][0]+tmp[2][1]=='DI':
            lab = 6
        elif tmp[2][0]+tmp[2][1]=='FE':
            lab = 0
    
        if tmp[2][2]=='3':
            imagesTs.append(tmpImg)
            labelsTs.append(lab)
        else:
            imagesTr.append(tmpImg)
            labelsTr.append(lab)
        
    imagesTr = np.array(imagesTr)  
    labelsTr = np.array(labelsTr)
    imagesTs = np.array(imagesTs)  
    labelsTs = np.array(labelsTs)
    return imagesTr,labelsTr,imagesTs ,labelsTs

         
if __name__ == '__main__':
    
    #STEP 0: Initialise constants and parameters
    inputSize = 64 * 64
    numClasses = 7
    
    

    lambdaa = 1e-4
    theta = 0.005 * np.random.randn(numClasses * inputSize)
    #STEP 1: Load data  
    imagesTr,labelsTr,imagesTs ,labelsTs = importFaces()
    #imagesTr, labelsTr=read([1,2,3,4,5,6,7,8,9,0], "training", ".") 
    #imagesTs, labelsTs=read([1,2,3,4,5,6,7,8,9,0], "testing", ".")
    labels = labelsTr
    print len(labels)
    #labels = np.array(labelsTr).flatten()    
    inputData = np.array(imagesTr).T   
    
    
    #======================================================================
    # STEP 2: Implement softmaxCost
    #
    #  Implement softmaxCost in softmaxCost.m.    
       
    cost ,grad = softmaxCost(theta, numClasses, inputSize, lambdaa, inputData, labels)
    #======================================================================
    # STEP 3: Gradient checking
    #
    #  As with any learning algorithm, you should always check that your
    #  gradients are correct before learning the parameters.
    #    
    #checkGradient()    
    #======================================================================
    # STEP 4: Learning parameters
    #
    #  Once you have verified that your gradients are correct, 
    #  you can start training your softmax regression code using softmaxTrain
    #  (which uses minFunc).

    #options.maxIter = 100
    options={'maxiter': 200, 'disp': True}
    softmaxModel = softmaxTrain(inputSize, numClasses, lambdaa,inputData, labels, options)
                          
    # Although we only use 100 iterations here to train a classifier for the 
    # MNIST data set, in practice, training for more iterations is usually
    # beneficial.
                          
    #======================================================================
    # STEP 5: Testing
    #
    #  You should now test your model against the test images.
    #  To do this, you will first need to write softmaxPredict
    #  (in softmaxPredict.m), which should return predictions
    #  given a softmax model and the input data.
    labels = labelsTs
    #labels = np.array(labelsTs).flatten()     
    inputData = np.array(imagesTs).T

    # You will have to implement softmaxPredict in softmaxPredict.m
    predictions= softmaxPredict(softmaxModel, inputData);

    print "Accuracy: {0:.2f}%".format(100 * np.sum(predictions == labels, dtype=np.float64) / labels.shape[0])

    # Accuracy is the proportion of correctly classified images
    # After 100 iterations, the results for our implementation were:
    #
    # Accuracy: 92.200%
    #
    # If your values are too low (accuracy less than 0.91), you should check 
    # your code for errors, and make sure you are training on the 
    # entire data set of 60000 28x28 training images 
    # (unless you modified the loading code, this should be the case)
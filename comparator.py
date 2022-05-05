#comparator
#import IWE_errors
import numpy as np
import keras
from keras.models import load_model
from comparator_helper import getFilePair, saveMapToImage, saveErrMapToImage, getAEMap, updateThresholds, getAvgAE, UTest
from imgPrep import addDChannel
#from naive_errors import simpleAngLoss, simpleAngErrorMetric
import tensorflow as tf
import os
#import sys

#allocating gpu memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

#imgDirs and normDirs must correspond, but can be changed according to which images you want to do the comparison on
imgDirs = {
    1: os.path.join(fileDirectory, 'cloth', 'images', 'Lc_left_edge')
    ,
    2: os.path.join(fileDirectory, 'hoody', 'images', 'Ll_front')
    ,
    3: os.path.join(fileDirectory, 'paper', 'images', 'Lc')
    ,
    4: os.path.join(fileDirectory, 'sweater', 'images', 'Ld_front')
    ,
    5: os.path.join(fileDirectory, 'tshirt', 'images', 'Lr_front_big')
    #33: '/home/quaku/textureless_deformable_surfaces/tshirt/images/Lr_tight/'
}

normDirs = {
    1: os.path.join(fileDirectory, 'cloth', 'normals', 'Lc_left_edge')
    ,
    2: os.path.join(fileDirectory, 'hoody', 'normals', 'Ll_front')
    ,
    3: os.path.join(fileDirectory, 'paper', 'normals', 'Lc')
    ,
    4: os.path.join(fileDirectory, 'sweater', 'normals', 'Ld_front')
    ,
    5: os.path.join(fileDirectory, 'tshirt', 'normals', 'Lr_front_big')
    #33: '/home/quaku/textureless_deformable_surfaces/tshirt/normals/Lr_tight/'
}

dirIDRanges = {
    1: [51, 1103]
    ,
    2: [48, 570]
    ,
    3: [25, 476]
    ,
    4: [26, 590]
    ,
    5: [132, 466]
    #33: [28, 525]
}

T = 1
#T is the number of images from each directory that we want to test on.
D = 5 #this is the number of directories we'll get test images from. Understand
#that we'll need to modify imgDirs, normDirs, and dirIDRanges according to this.
DIM = 224

#load both models as PWE and IWE, respectively.
PWE = load_model('pixel_wise_sne.h5')
IWE = load_model('image_wide_sne.h5')
#get them to predict. actually do a test cuz that will give you the metrics
#immediately. What we want to get is the average angular error, and we want to
#count how many times the avg angular error is above 10degs, 20degs, and 30degs.
#The average per image, i.e. per image angular error. 

#now to load filePairs. We'll load the last image in each directory. Or the last
#n images.
PWEThresholds = {11: 0, 22: 0, 45: 0, 90: 0, 180: 0}
IWEThresholds = {11: 0, 22: 0, 45: 0, 90: 0, 180: 0}
PWETot = 0
IWETot = 0
scores = []
for d in range(1, D+1): #looping through directories
    for t in range(T): #looping through reverse image IDs
        #assume we have selected a file pair. call it Pair. index 0 holds the image.
        pair = getFilePair(imgDirs, normDirs, dirIDRanges, d, t)
        PWEMap = np.zeros((DIM, DIM, 3))
        IWEMap = np.zeros((DIM, DIM, 3))
        for x in range(DIM):
            for y in range(DIM):
                #adding a d-channel and RGB scaling. But only if [x,y] is alive
                #in normMap.
                if not((pair[1][x, y] == [0, 0, 0]).all()):
                    PWEimg = addDChannel(pair[0], [x, y], DIM)
                    PWEimg = np.expand_dims(PWEimg, axis=0)
                    pred = PWE(PWEimg)
                    pred = tf.make_ndarray(pred)
                    pred = np.reshape(pred, (3))
                    PWEMap[x, y] = pred
                    PWEMap[x, y] /= np.sqrt(np.power(PWEMap[x,y,0], 2) + np.power(PWEMap[x,y,1], 2) + np.power(PWEMap[x,y,2], 2)) # do this when you're done looping
                    #that was to normalize the predicted normal vector
                #call PWE and add the result to PWEMap[x, y]
        AEMapForPWE = getAEMap(pair[1], PWEMap)
        #count threshold breaks
        updateThresholds(AEMapForPWE, PWEThresholds, pair[1])
        PWEScore = getAvgAE(AEMapForPWE, pair[1])
        scores.append(['PWE', PWEScore])
        PWETot += PWEScore
        #save the maps to disk
        saveMapToImage(PWEMap, pair[1], d, t, 'pixel_wise')
        saveErrMapToImage(AEMapForPWE, pair[1], d, t, 'pixel_wise')

        IWEimg = pair[0]
        IWEimg = np.expand_dims(IWEimg, axis=0)
        IWEMap = IWE(IWEimg)
        IWEMap = tf.make_ndarray(IWEMap)
        IWEMap = np.reshape(IWEMap, (DIM, DIM, 3))
        for x in range(DIM):
            for y in range(DIM):
                if (pair[1][x, y] == [0, 0, 0]).all():
                    IWEMap[x, y] = [0, 0, 0]
                else:
                    IWEMap[x,y] /= np.sqrt(np.power(IWEMap[x,y,0], 2) + np.power(IWEMap[x,y,1], 2) + np.power(IWEMap[x,y,2], 2))
                    #that was to normalize the predicted normal vector
        AEMapForIWE = getAEMap(pair[1], IWEMap)
        #count threshold breaks
        updateThresholds(AEMapForIWE, IWEThresholds, pair[1]) #write the thresholds to file when all is done
        IWEScore = getAvgAE(AEMapForIWE, pair[1])
        scores.append(['IWE', IWEScore])
        #save the maps to file
        saveMapToImage(IWEMap, pair[1], d, t, 'image_wide')
        saveErrMapToImage(AEMapForIWE, pair[1], d, t, 'image_wide')

        winner, U = UTest(scores)
        print('winner: ' + winner + ' U: ' + str(U))

#guidelines for programming the comparison:
#compute estimated normal map array. The P array.
    #we'll load the models, PWE and IWE.
    #we'll load a file pair filePair.
    #we'll get each model to do a prediction, i.e. model(x, training=False).
#Then compute angular error AE for that array. Call it the AE array. 
#Then generate the GT nmap, the estimated nmap, and the AE map for each
#prediction. We should save these images to files.

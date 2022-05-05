#IWE_reboot
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, UpSampling2D
#import numpy as np
import IWE_helper
#import errors
import pickle
from grapher import graphTraining, graphTesting
import tensorflow as tf
from math import floor
from datetime import datetime
import os
#import sys

#allocating gpu memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

#imgDirs contains directories in which images reside
imgDirs = {
    1: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Lc_left_edge'),
    2: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Lc_tl_te_corns'),
    3: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Lc_top_edge_a'),
    4: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Lc_top_edge_b'),
    5: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Ld_bottom_edge'),
    6: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Ld_top_edge'),
    7: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Ll_bottom_edge'),
    8: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Ll_left_edge'),
    9: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Ll_tl_tl_corns'),
    10: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Ll_top_edge'),
    11: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Lr_bottom_edge'),
    12: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Lr_left_edge'),
    13: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Lr_top_edge_1')
    ,
    14: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'hoody', 'images', 'Ll_front')
    ,
    15: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'paper', 'images', 'Lc'),
    16: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'paper', 'images', 'Ll'),
    17: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'paper', 'images', 'Lr')
    ,
    18: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'images', 'Lc_front'),
    19: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'images', 'Ld_front'),
    20: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'images', 'Ll_front'),
    21: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'images', 'Lr_front')
    ,
    22: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Lc_front'),
    23: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Ld_back'),
    24: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Ld_front'),
    25: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Ld_front_big'),
    26: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Ld_tight'),
    27: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Ll_back'),
    28: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Ll_front'),
    29: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Ll_tight'),
    30: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Lr_back'),
    31: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Lr_front'),
    32: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Lr_front_big')
}

#normDirs contains directories in which normal maps reside
normDirs = {
    1: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Lc_left_edge'),
    2: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Lc_tl_te_corns'),
    3: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Lc_top_edge_a'),
    4: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Lc_top_edge_b'),
    5: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Ld_bottom_edge'),
    6: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Ld_top_edge'),
    7: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Ll_bottom_edge'),
    8: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Ll_left_edge'),
    9: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Ll_tl_tl_corns'),
    10: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Ll_top_edge'),
    11: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Lr_bottom_edge'),
    12: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Lr_left_edge'),
    13: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'normals', 'Lr_top_edge_1')
    ,
    14: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'hoody', 'normals', 'Ll_front')
    ,
    15: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'paper', 'normals', 'Lc'),
    16: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'paper', 'normals', 'Ll'),
    17: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'paper', 'normals', 'Lr')
    ,
    18: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'normals', 'Lc_front'),
    19: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'normals', 'Ld_front'),
    20: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'normals', 'Ll_front'),
    21: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'normals', 'Lr_front')
    ,
    22: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Lc_front'),
    23: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Ld_back'),
    24: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Ld_front'),
    25: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Ld_front_big'),
    26: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Ld_tight'),
    27: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Ll_back'),
    28: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Ll_front'),
    29: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Ll_tight'),
    30: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Lr_back'),
    31: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Lr_front'),
    32: os.path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'normals', 'Lr_front_big')
}

#dirIDRanges specifies valid ranges of files in normDirs and imgDirs
dirIDRanges = {
    1: [51, 1103],
    2: [75, 1314],
    3: [67, 547],
    4: [548, 1027],
    5: [105, 826],
    6: [78, 1064],
    7: [103, 1166],
    8: [182, 1135],
    9: [59, 1028],
    10: [206, 1063],
    11: [156, 1149],
    12: [52, 1068],
    13: [41, 773]
    ,
    14: [48, 570]
    ,
    15: [25, 476],
    16: [35, 468],
    17: [117, 421]
    ,
    18: [23, 563],
    19: [26, 590],
    20: [24, 583],
    21: [22, 562]
    ,
    22: [93, 844],
    23: [54, 686],
    24: [47, 625],
    25: [37, 691],
    26: [39, 587],
    27: [67, 713],
    28: [136, 723],
    29: [84, 651],
    30: [117, 662],
    31: [42, 589],
    32: [132, 466]
    #33: [28, 525]
}

#reloading model
model = load_model('image_wide_sne.h5')

#reloading tarining losses
with open('trainLoss_IWE.data', 'rb') as trainLossIn:
    trainLosses = pickle.load(trainLossIn)

#reloading testlosses
with open('testLoss_IWE.data', 'rb') as testLossIn:
    testLosses = pickle.load(testLossIn)

last = trainLosses[-1] #the last line of trainLosses
steps = last[0] 
epochs = floor(steps/125) #we take two files from each dir in each iteration

#EPOCHS = 10 #for now, we might change it.
#steps = 0
#epochs = 0
remainders = IWE_helper.genRemainders(dirIDRanges) #the indices of files in imgDirs and normDirs that have not been trained on in this epoch
stop = False
pauseTimes = {1: '10', 2: '12', 3: '14', 4: '16', 5: '18', 6: '20', 7: '22', 8: '00'} #these are the hours at which the training will pause and the user will decide whether to cease training
#pauseTimes = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '0'}
pauseIndx = 2
progressPath = os.path.join(fileDirectory, 'IWE Progress') #path to which progress graphs will be saved

while stop == False:
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    steps += 1
    #generate a training batch.
    Xtrain, Ytrain = IWE_helper.genTrainBatch(imgDirs, normDirs, remainders)
    Xtrain = tf.convert_to_tensor(Xtrain)
    Ytrain = tf.convert_to_tensor(Ytrain)
    trainLoss = model.train_on_batch(Xtrain, y=Ytrain)

    if steps == 1:
        trainLosses.append([steps, trainLoss])

    if steps%5 == 0:
        trainLosses.append([steps, trainLoss])
        print('steps: ' + str(steps) + ', loss: ' + str(trainLoss) + ', epochs: ' + str(epochs))

    if steps%50 == 0:
        graphTraining(trainLosses, steps, progressPath)
    #train on batch here
    #after k steps generate a test batch and then run a test.
    if len(remainders[1]) == 0:
        remainders = IWE_helper.genRemainders(dirIDRanges)
        epochs += 1
        Xtest, Ytest = IWE_helper.genTestBatch(dirIDRanges, imgDirs, normDirs)
        Xtest = tf.convert_to_tensor(Xtest)
        Ytest = tf.convert_to_tensor(Ytest)
        testLoss = model.test_on_batch(Xtest, y=Ytest)
        testLosses.append([steps, testLoss])
        graphTesting(testLosses, steps, progressPath)
        #test on batch here
        #save progress here. We don't need to write remainders cuz if we get
        #here, remainders will be empty
        model.save('image_wide_sne.h5')
        print('saved model to disk')

        trainLossOut = open('trainLoss_IWE.data', 'wb')
        pickle.dump(trainLosses, trainLossOut)
        trainLossOut.close()

        testLossOut = open('testLoss_IWE.data', 'wb')
        pickle.dump(testLosses, testLossOut)
        testLossOut.close()

    now = datetime.now()
    currentTime = now.strftime('%H:%M:%S')
    nowHour = currentTime[0:2] #nowHour equals the current hour
    if (nowHour == pauseTimes[pauseIndx]):
        #save progress here
        model.save('image_wide_sne.h5')
        print('saved model to disk')


        trainLossOut = open('trainLoss_IWE.data', 'wb')
        pickle.dump(trainLosses, trainLossOut)
        trainLossOut.close()

        testLossOut = open('testLoss_IWE.data', 'wb')
        pickle.dump(testLosses, testLossOut)
        testLossOut.close()

        pause = input('Do you want to halt training? (y/n): ')
        if pause == 'y':
            stop = True
        else:
            pauseIndx += 1
            if pauseIndx > 8:
                pauseIndx = 1

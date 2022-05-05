#pixel-wise predictor with chosen set
#import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization
import numpy as np
#import matplotlib.pyplot as plt
#import random
from imgPrep_smarter_choice import genTrainBatch, genTestBatch
from datetime import datetime
import pickle
#import PWE_errors
from grapher import graphTraining, graphTesting
import tensorflow as tf
import os
#import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
#we need to put all the directory addresses in a dict.
#we'll also have to record what the maximum ID is in each
#directory
#so we'll first create the dataset dict. We'll want to be able
#to randomly choose the dir from which to choose the training
#image
#imgDirs contains the directories in which the images reside
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

#normDirs contains the directories in which the ground truth normal maps reside
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

#dirIDRanges contains the ranges of valid file indices in imgDirs and normDirs
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

DIM = 224

#define model
model = Sequential()
model.add(Conv2D(12, kernel_size=(3,3), activation='relu', use_bias=False,
input_shape=(DIM, DIM, 4)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(12, kernel_size=(4, 4), activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(16, activation='relu', use_bias=False))
model.add(BatchNormalization())
model.add(Dense(3, activation='tanh'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.CosineSimilarity())

#now to make the arrays to hold the the pixel ids in the training images
#do this part with np.arrange
commonPixSet = np.zeros((224, 224, 2), int) 
for x in range(0, DIM):
    for y in range(0, DIM):
        commonPixSet[x, y] = [x, y] #commonPixSet is simply the set of all pixels in a 224x224 image

steps = 0
epochs = [0, 0, 0, 0, 0]
clothChosen = set()
hoodyChosen = set()
paperChosen = set()
sweaterChosen = set()
tshirtChosen = set()
chosenSets = [clothChosen, hoodyChosen, paperChosen, sweaterChosen, tshirtChosen] #elements of these sets are the pixels from specified directories that have been trained on
trainLosses = []
testLosses = []
stop = False
pauseTimes = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '0'} #10 minute intervals between pauses
pauseIndx = 5 #this is the index of the next time the user wishes to pause at
progressPath = os.path.join(fileDirectory, 'PWE Progress')

while stop == False:
    steps += 1
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    #the real issue is ensuring that each pixel position is chosen a reasonable
    #amount of times. For that, surely a uniform random distribution should work
    #well enough. We should remove a pixel from the set S of elegible pixels once
    #it has been selected for training. S should be replenished when it is
    #completely depleted. But S will never be fully depleted because we never
    #choose black pixels. S is replenished when the there are no live pixels in
    #the set [currentImageNormalMap] - S

    Xtrain, Ytrain, epochs = genTrainBatch(dirIDRanges, imgDirs, normDirs, DIM, commonPixSet, chosenSets, epochs)
    Xtrain = tf.convert_to_tensor(Xtrain)
    Ytrain = tf.convert_to_tensor(Ytrain)
    loss = model.train_on_batch(Xtrain, y=Ytrain)
    #every n iterations, run the model on the validation data.
    if steps == 1:
        trainLosses.append([steps, loss])

    if steps%5 == 0:
        trainLosses.append([steps, loss])
        #output the loss.
        #and save trainLoss to file - overwriting it.
        graphTraining(trainLosses, steps, progressPath)
        print('steps: ' + str(steps) + ', loss: ' + str(loss) + ', epochs: ' + str(epochs))

    if steps%50 == 0:
        Xtest, Ytest = genTestBatch(DIM, commonPixSet)
        Xtest = tf.convert_to_tensor(Xtest)
        Ytest = tf.convert_to_tensor(Ytest)
        testLoss = model.test_on_batch(Xtest, y=Ytest)
        testLosses.append([steps, testLoss])
        #I think we'd also like to display the training graph here.
        graphTesting(testLosses, steps, progressPath)
        #choose the last ten images from every dir to test on. They should be
        #the last ten images in a dir.
        #loop through the list of file pairs and prepare the X_test and Y_test
        #arrays. Actually, we'll just test on 1 image from each of the four
        #conditions - and even then only on some of the pixels therein
            #use scaleRGB, stripBlackPixs.
            #randomly choose a pixel p from pixSet.
            #then use p to add a Dchannel to the test image.
            #add the newImage to X_test, and add the counterpart of p in the normal file
            #to Y_test - make sure to first abs the counterpart
    #check time. Put it in a string T.
    now = datetime.now()
    currentTime = now.strftime('%H:%M:%S')
    nowHour = currentTime[3]
    if (nowHour == pauseTimes[pauseIndx]):
        #save progress here
        model.save('pixel_wise_sne.h5')
        print('saved model to disk')

        chosenOut = open('chosen_PWE.data', 'wb')
        pickle.dump(chosenSets, chosenOut)
        chosenOut.close()

        trainLossOut = open('trainLoss_PWE.data', 'wb')
        pickle.dump(trainLosses, trainLossOut)
        trainLossOut.close()

        testLossOut = open('testLoss_PWE.data', 'wb')
        pickle.dump(testLosses, testLossOut)
        testLossOut.close()

        epochsOut = open('epochs.data', 'wb')
        pickle.dump(epochs, epochsOut)
        epochsOut.close()

        pause = input('Do you want to halt training? (y/n): ')
        if pause == 'y':
            stop = True
        else:
            pauseIndx += 1
            if pauseIndx > 6:
                pauseIndx = 1


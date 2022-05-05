#sne_image_wide
#import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, UpSampling2D
#import numpy as np
#from datetime import datetime
from IWE_helper import genRemainders, genTestBatch, genTrainBatch
#import errors
import pickle
from grapher import graphTraining, graphTesting
import tensorflow as tf
import os
#import sys

#allocating gpu memory here
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

#normDirs contains directories in which the ground truth normal maps reside
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

#dirIDRanges contains legal ranges for files in imgDirs and normDirs
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

DIM = 224 #DIM is size of square images

#for every batch, we'll want equal representation from each
#lowest-level-directory.
#we'll train on 250 images from each such directory. If we use a batch size of
#64 (taking 2 images from each directory) then each epoch will take 125 iterations
#of the training loop.
#should we randomly select the images from each directory? If we did, we'd have
#to keep track of which indices have already been selected from each directory
#in each epoch.
#we'll choose the files for the batch, then put them in X and Y, then shuffle
#them.

#define the model here.
model = Sequential()
model.add(Conv2D(15, (3, 3), activation='relu', padding='same', use_bias=False,
input_shape=(DIM, DIM, 3)))
model.add(BatchNormalization())
model.add(AveragePooling2D((4, 4), padding='same'))
model.add(Conv2D(30, (3, 3), activation='relu', padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(UpSampling2D((4, 4)))
model.add(Conv2D(15, (3, 3), activation='relu', padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Conv2D(3, (3, 3), activation='tanh', padding = 'same'))
#we need to make sure that this model produces output of the right shape/size.
#No, it definitely will cuz it only uses same padding
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.cosine_similarity)

EPOCHS = 1 #number of epochs, i.e. number of times the model is trained on the whole training set
steps = 0
epochs = 0
trainLosses = []
testLosses = []
remainders = genRemainders(dirIDRanges) #the indices of files in imgDirs and normDirs that have not been trained on in this epoch
progressPath = os.path.join(fileDirectory, 'IWE Progress')

while epochs < EPOCHS:
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    steps += 1
    #generate a training batch.
    Xtrain, Ytrain = genTrainBatch(imgDirs, normDirs, remainders)
    Xtrain = tf.convert_to_tensor(Xtrain)
    Ytrain = tf.convert_to_tensor(Ytrain)
    trainLoss = model.train_on_batch(Xtrain, y=Ytrain)

    if steps == 1:
        trainLosses.append([steps, trainLoss])
        
    #output current training loss now
    if steps%100 == 0:
        trainLosses.append([steps, trainLoss])
        print('steps: ' + str(steps) + ', loss: ' + str(trainLoss))

    if steps%1000 == 0:
        graphTraining(trainLosses, steps, progressPath)
    #train on batch here
    #after k steps generate a test batch and then run a test.
    if len(remainders[1]) == 0:
        remainders = genRemainders(dirIDRanges)
        epochs += 1
        Xtest, Ytest = genTestBatch(dirIDRanges, imgDirs, normDirs)
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
    

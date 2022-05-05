#image_wide helper
import numpy as np
import random
import matplotlib.pyplot as plt
from os import path
#import sys #I included this because I thought we needed for generating file paths

def genFilePairs(imgDirs, normDirs, remainders):
    #what we should do is create a list of pairs, where each pair is an image
    #file location and a normal file location. Then we'll shuffle the list. Then
    #we'll generate the training data from the file names.
    filePairs = []
    for k in range(1, 33):
        #randomly select two indices from remainders[k].
        #generate the filenames for these indices, using normDirs and imgDirs.
        #append each pair of filenames to filePairs.
        filesExist = False
        p = random.randint(0, len(remainders[k])-1)
        index = remainders[k][p]
        while not(filesExist):
            imgPath1, normPath1 = genFilePaths(k, imgDirs, normDirs, index)
            if (path.exists(imgPath1) and path.exists(normPath1)):
                filesExist = True
                del remainders[k][p]
            else:
                index = index-1
        imgPath1, normPath1 = genFilePaths(k, imgDirs, normDirs, index)

        filesExist = False
        q = random.randint(0, len(remainders[k])-1)
        index = remainders[k][q]
        while not(filesExist):
            imgPath2, normPath2 = genFilePaths(k, imgDirs, normDirs, index)
            if (path.exists(imgPath2) and path.exists(normPath2)):
                filesExist = True
                del remainders[k][q]
            else:
                index = index-1
        imgPath2, normPath2 = genFilePaths(k, imgDirs, normDirs, index)

        filePairs.append([imgPath1, normPath1])
        filePairs.append([imgPath2, normPath2])
    random.shuffle(filePairs)
    return filePairs

def genFilePaths(dirID, imgDirs, normDirs, index):
    #this generates the img and normal filenames for a given index and dirID
    if index >= 1000:
        #indxStr must be just the string version of index
        indxStr = str(index)
    elif index >= 100:
        indxStr = '0' + str(index)
    elif index >= 10:
        indxStr = '00' + str(index)
    else:
        indxStr = '000'+str(index)

    imgName = 'rgb_'  + indxStr + '.tiff'
    normFileName = 'normals_' + indxStr + '.npz'
    imgPath = path.join(imgDirs[dirID], imgName)
    normPath = path.join(normDirs[dirID], normFileName)
    return imgPath, normPath

def genTrainBatch(imgDirs, normDirs, remainders):
    Xtrain = []
    Ytrain = []
    filePairs = genFilePairs(imgDirs, normDirs, remainders)
    for pair in filePairs:
        img = plt.imread(pair[0])
        img = np.asarray(img, dtype=np.float32)
        scaleRGB(img)
        nmap = np.load(pair[1])['normals']
        Xtrain.append(img)
        Ytrain.append(nmap)

    return Xtrain, Ytrain

def genTestBatch(dirIDRanges, imgDirs, normDirs):
    Xtest = []
    Ytest = []
    for dir in range(1, 33):
        index = dirIDRanges[dir][1]
        imgPath, normPath = genFilePaths(dir, imgDirs, normDirs, index)
        img = plt.imread(imgPath)
        img = np.asarray(img, dtype=np.float32)
        scaleRGB(img)
        nmap = np.load(normPath)['normals']
        Xtest.append(img)
        Ytest.append(nmap)

    return Xtest, Ytest

def genRemainders(dirIDRanges):
    #this function should be called at the start of every epoch, cuz
    #remainders needs to be restored at the start of an epoch
    remainders = {1: [],
                2: [],
                3: [],
                4: [],
                5: [],
                6: [],
                7: [],
                8: [],
                9: [],
                10: [],
                11: [],
                12: [],
                13: [],
                14: [],
                15: [],
                16: [],
                17: [],
                18: [],
                19: [],
                20: [],
                21: [],
                22: [],
                23: [],
                24: [],
                25: [],
                26: [],
                27: [],
                28: [],
                29: [],
                30: [],
                31: [],
                32: []
    }
    #this populates the remainders dict. It should take up 64KBs.
    #we want to use the first 250 files in each directory. Note the file indices
    #in the kth directory start from dirIDRanges[k][0]
    for k in range(1, len(dirIDRanges)+1):
        for i in range(250):
            remainders[k].append(dirIDRanges[k][0]+i)
    return remainders

def scaleRGB(img):
    #img = img.astype('float32')
    img /= 255

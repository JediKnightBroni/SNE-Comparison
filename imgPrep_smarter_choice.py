#imgPrep with smart choice
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from os import path
#import sys

def addDChannel(img, pixel, dim):
    #img is the image on which we are training
    #pixelX and pixelY are the  x and y coords of the pixel of interest
    #dim is the x and y dimensions of the square image. The image must be square
    pixelX = pixel[0]
    pixelY = pixel[1]
    xStart = 0-pixelX
    xEnd = dim-pixelX #dim = 224 instead of 223 cuz mgrid stop index is non-inclusive
    yStart = 0-pixelY
    yEnd = dim-pixelY
    maxD = math.sqrt(math.pow((0-dim), 2)+math.pow((0-dim), 2))
    grids = np.mgrid[xStart:xEnd, yStart:yEnd]
    Dchannel = np.sqrt(np.add(np.power(grids[0], 2), np.power(grids[1], 2)))
    Dchannel = np.divide(Dchannel, maxD)
    newImg = np.dstack((img, Dchannel))
    return newImg

def stripBlackPixs(normFile, pixSet):
#    newPixSet = copy.deepcopy(pixSet)
    #deepcopy is super slow. Get rid of it.
    liveSet = pixSet[normFile.any(axis=-1)] #liveset is the set ofall pixels whose normals are not (0,0,0)
    return liveSet

def stripChosen(pixSet, chosen, epochs):
    newPixSet = pixSet.difference(chosen)
    if len(newPixSet) == 0:
        chosen.clear() #sets are passed by reference cuz they're mutable
        epochs += 1
        x = next(iter(pixSet))
        newPixSet.add(x)
    #if the set subtraction did not empty newPixSet, then we want to use
    #newPixSet as pixSet. If it did empty newPixSet, then we want to use the
    #PixSet that was originally supplied to this function - we obviously don't
    #want to try to sample an empty pixSet.
    return newPixSet, epochs

def scaleRGB(img):
    img /= 255

def genFilePairs(dirIDRanges, imgDirs, normDirs):
    #returns a list of pairs of the form [imagefile, normalfile]
    dir = 1 #this is the id of the directory we are currently in
    typeID = 0 #this is an id for the type of image we are adding to filePairs. types are sweater, cloth etc.
    filePairs = []
    for id in imgDirs:
        #randomly generate a number within the pertinent dirIDRanges.
        #generate the string for the normal and image file.
        #load arrays of the image file and the normal file into memory
        #append the pair of arrays into filePairs.
        #So we can do tests with the last ten images in a directory
        filesExist = False
        index = random.randint(dirIDRanges[id][0], dirIDRanges[id][1]-10)
        while not(filesExist):
            imgPath, normPath = genFilePaths(id, imgDirs, normDirs, index)
            if (path.exists(imgPath) and path.exists(normPath)):
                filesExist = True
            else:
                index = index-1

        #load the image and normal files into numpy arrays
        img = plt.imread(imgPath)
        #convert img to numpy array
        img = np.asarray(img, dtype=np.float32)
        scaleRGB(img)
        nmap = np.load(normPath)['normals']
        if dir <= 13:
            typeID = 0
        elif dir == 14:
            typeID = 1
        elif dir <= 17:
            typeID = 2
        elif dir <= 21:
            typeID = 3
        elif dir <= 32:
            typeID = 4
        filePairs.append([img, nmap, typeID])
        #we should shuffle file pairs here
        dir += 1
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
    normPath = path.join(normDirs[dirID] + normFileName)
    return imgPath, normPath

def genTestBatch(dim, commonPixSet):
    #select the four images. they should be Lc, Ll, Ld, Lr.
    #indxList = [1, 14, 16, 19, 32] #1103, 570, 468, 590, 466
    absolutepath = path.abspath(__file__)
    fileDirectory = path.dirname(absolutepath)
    imgList = []
    normList = []
    Xtest = []
    Ytest = []

    imgList.append(path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Lc_left_edge', 'rgb_1103.tiff'))
    imgList.append(path.join(fileDirectory, 'textureless_deformable_surfaces', 'hoody', 'images', 'Ll_front', 'rgb_0570.tiff'))
    #imgList.append('/home/quaku/textureless_deformable_surfaces/paper/images/Ll/rgb_0468.tiff')
    imgList.append(path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'images', 'Ld_front', 'rgb_0590.tiff'))
    imgList.append(path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Lr_front_big', 'rgb_0466.tiff'))

    normList.append(path.join(fileDirectory, 'textureless_deformable_surfaces', 'cloth', 'images', 'Lc_left_edge', 'normals_1103.npz'))
    normList.append(path.join(fileDirectory, 'textureless_deformable_surfaces', 'hoody', 'images', 'Ll_front', 'normals_0570.npz'))
    #normList.append('/home/quaku/textureless_deformable_surfaces/paper/normals/Ll/normals_0468.npz')
    normList.append(path.join(fileDirectory, 'textureless_deformable_surfaces', 'sweater', 'images', 'Ld_front', 'normals_0590.npz'))
    normList.append(path.join(fileDirectory, 'textureless_deformable_surfaces', 'tshirt', 'images', 'Lr_front_big', 'normals_0466.npz'))
    #we should test on 64 pixels. and we should call this function every time
    #we want to test, otherwise we won't test on all the pixels.

    #now to choose the pixels. first strip black and strip chosen.
    #we'll choose 16 pixels from each image in imgList
    for k in range(4):
        img = plt.imread(imgList[k])
        img = np.asarray(img, dtype=np.float32)
        scaleRGB(img)
        nmap = np.load(normList[k])['normals']

        pixList = stripBlackPixs(nmap, commonPixSet)
        pixList.ravel()
        for l in range(16):
            p = random.randint(0, len(pixList)-1)
            curPix = pixList[p]
            imgForX = addDChannel(img, curPix, dim)
            targNorm = nmap[curPix[0], curPix[1]]
            Xtest.append(imgForX)
            Ytest.append(targNorm)

    return Xtest, Ytest

def genTrainBatch(dirIDRanges, imgDirs, normDirs, dim, pixSet, chosenSets, epochs):
    filePairs = genFilePairs(dirIDRanges, imgDirs, normDirs)
    Xtrain = []
    Ytrain = []
    for pair in filePairs:
        #create a pixlist of all pixels in the image file. pixList must be a set.
        #strip the black pixels from the pixlist using stripBlackPixs.
        typeID = pair[2]
        liveArray = stripBlackPixs(pair[1], pixSet) #liveSet is an array though
        liveTuples = tuple(map(tuple, liveArray)) #because you can't add arrays to sets
        liveSet = set(liveTuples)
        #strip the pixlist of all pixels in Chosen with stripChosen.
        spinsters, epochs[typeID] = stripChosen(liveSet, chosenSets[typeID], epochs[typeID]) #spinsters is the set of pixels in liveSet that have not been chosen yet
            #if pixlist is empty now, empty out Chosen and return the pixList
            #originally submitted to stripChosen.
        #randomly choose a pixel p from pixlist and add it to Chosen.
        spinsterList = list(spinsters) #this is so that we can randomly select a pixel in spinsterList
        p = random.randint(0, len(spinsterList)-1)
        curPix = spinsterList[p]
        chosenSets[typeID].add(curPix)
        #curPix is the pixel of interest.
        #compute the D-channel with p as the pixel of interest and add the
        #channel to the image file to make imgWithD.
        img = addDChannel(pair[0], curPix, dim)
        targNorm = pair[1][curPix[0], curPix[1]]
        #add imgWithD to the training batch X_train, and add the absed ise of p
        #in the normal file to Y_train.
        Xtrain.append(img)
        Ytrain.append(targNorm)

    return Xtrain, Ytrain, epochs

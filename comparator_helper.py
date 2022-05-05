#comparator_helper
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from naive_errors import simpleAngLoss
import os
#import sys

def getFilePair(imgDirs, normDirs, dirIDRanges, dirID, revImageID):
    #first generate the actual file names.
    last = dirIDRanges[dirID][1]
    index = last - revImageID
    if index >= 1000:
        #indxStr must be just the string version of index
        indxStr = str(index)
    elif index >= 100:
        indxStr = '0' + str(index)
    elif index >= 10:
        indxStr = '00' + str(index)
    else:
        indxStr = '000'+str(index)

    imgFileName = 'rgb_'  + indxStr + '.tiff'
    normFileName = 'normals_' + indxStr + '.npz'
    imgPath = os.path.join(imgDirs[dirID], imgFileName)
    normPath = os.path.join(normFileName[dirID], normFileName)

    img = plt.imread(imgPath)
    #convert img to numpy array
    img = np.asarray(img, dtype=np.float32)
    scaleRGB(img)
    nmap = np.load(normPath)['normals']
    pair = []
    pair.append(img)
    pair.append(nmap)
    return pair

def scaleRGB(img):
    img /= 255

def saveMapToImage(map, GTMap, d, t, model):
    #saves normal map to image. GTMap is needed to determine dead pixels
    #model is a string - the name of the model
    data = np.zeros((224, 224, 3))
    data = np.absolute(np.subtract(map, 1))
    data = np.floor(np.multiply(data, 127.5))
    dead = np.all(GTMap == [0, 0, 0], axis = -1)
    rs, cs = dead.nonzero()
    data[rs, cs, :] = [0, 0, 0]
    img = Image.fromarray(np.uint8(data), 'RGB')
    absolutepath = os.path.abspath(__file__)
    fileDirectory = os.path.dirname(absolutepath)
    imgFileName = 'dir_' + str(d) + '_' + str(t) + '_' + model + '_map'+ '.png'
    imgFilePath = os.path.join(fileDirectory, 'images', imgFileName) #the path to the normal map image
    img.save(imgFilePath) #we should save the images in a special purpose dir

def saveErrMapToImage(errMap, GTMap, d, t, model):
    #saves map of angular errors to an image. GTMap is needed to determine dead pixels
    data = np.zeros((224, 224, 3))
    data += 180
    degErrMap = np.degrees(errMap)
    degErrMap = np.reshape(degErrMap, (224, 224, 1))
    data [:, :, 1:] -= degErrMap
    data = np.floor(np.multiply(data, 255/180))
    dead = np.all(GTMap == [0, 0, 0], axis = -1)
    rs, cs = dead.nonzero()
    data[rs, cs, :] = [0, 0, 0]
    img = Image.fromarray(np.uint8(data), 'RGB')
    absolutepath = os.path.abspath(__file__)
    fileDirectory = os.path.dirname(absolutepath)
    imgFileName = 'dir_' + str(d) + '_' + str(t) + '_' + model + '_AEMap'+'.png'
    imgFilePath = os.path.join(fileDirectory, 'images', imgFileName) #the path to the AE image
    img.save(imgFilePath)

def getAvgAE(AEMap, GTMap):
    #calculates average angular error of AEMap. GTMap is needed to determine dead pixels
    AEMapInDegrees = np.degrees(AEMap)
    totalAE = 0
    preds = 0
    for x in range(224):
        for y in range(224):
            if not((GTMap[x, y] == [0, 0, 0]).all()):
                totalAE += AEMapInDegrees[x, y]
                preds += 1

    avg = totalAE/preds
    return avg, preds

def getTotPreds(totPredsList):
    tot = 0
    for k in range(len(totPredsList)):
        tot += totPredsList[k]
    return tot

def getDiffs(AEMap, GTMap, diffs):
    AEMapInDegrees = np.degrees(AEMap)

    for x in range(224):
        for y in range(224):
            if not((GTMap[x, y] == [0, 0, 0]).all()):
                diffs.append(AEMapInDegrees[x, y])

def getStanDev(diffs, mean, preds):
    result = np.sqrt(np.divide(np.sum(np.power(np.subtract(diffs, mean), 2)), preds)) 
    return result

def getFinalAvg(averages, totPredsList):
    totPreds = 0
    weights = []
    average = 0
    for k in range(len(totPredsList)):
        totPreds += totPredsList[k]

    for j in range(len(totPredsList)):
        weightJ = totPredsList[j]/totPreds
        weights.append(weightJ)

    for i in range(len(totPredsList)):
        average += averages[i]*weights[i]

    return average

def getAEMap(GTMap, predMap):
    #generates angular error map
    #GTMap is the ground truth normal map.
    #predMap is the normal map predicted by the model.
    #AEMap is the array of angular errors twx GTArray and predArray.
    DIM = 224
    AEMap = np.zeros((DIM, DIM))
    AEMap = simpleAngLoss(GTMap, predMap)
    dead = np.all(GTMap == [0, 0, 0], axis = -1)
    rs, cs = dead.nonzero()
    AEMap[rs, cs] = 0
    return AEMap


def updateThresholds(AEMap, thresholds, GTMap):
    for x in range(224):
        for y in range(224):
            if not((GTMap[x, y] == [0, 0, 0]).all()):
                if AEMap[x, y] < math.radians(15):
                    thresholds[15] += 1
                elif AEMap[x, y] < math.radians(30):
                    thresholds[30] += 1
                elif AEMap[x, y] < math.radians(45):
                    thresholds[45] += 1
                elif AEMap[x, y] < math.radians(60):
                    thresholds[60] += 1
                else:
                    thresholds[180] += 1

def getFractions(thresholds, fractions):
    #assigns to keys of fractions the percentage of normal predictions within the accuracy denoted by the key
    total = 0
    total += thresholds[15]
    total += thresholds[30]
    total += thresholds[45]
    total += thresholds[60]
    total += thresholds[180]

    fractions[15] = thresholds[15]/total
    fractions[30] = thresholds[30]/total
    fractions[45] = thresholds[45]/total
    fractions[60] = thresholds[60]/total
    fractions[180] = thresholds[180]/total


def UTest(scores):
    #compares PWE and IWE according to the Mann-Whitney U-test
    #we don't need to count Ni and Np. Each is half of len(scores)
    sortScores(scores)
    Ni = len(scores)/2
    Np = Ni
    Nx = Ni
    PWERankTot = countRankTot('PWE', scores)
    IWERankTot = countRankTot('IWE', scores)
    if PWERankTot > IWERankTot:
        winner = 'PWE'
        U = Ni * Np + Nx * (Nx+1)/2 - PWERankTot
    elif IWERankTot > PWERankTot:
        winner = 'IWE'
        U = Ni * Np + Nx * (Nx+1)/2 - IWERankTot
    else:
        winner = 'equal'
        U = 1000

    return winner, U #winner with U-value

def sortScores(scores):
    #return nothing. just modify scores.
    #we'll use selection sort.
    #it will rank the scores in order of increasing accuracy.
    for i in range(len(scores)):
        maxIdx = i
        for j in range(i+1, len(scores)):
            #after this loop the maxIdx will be the index of the worst score.
            if scores[maxIdx, 1] < scores[j, 1]:
                maxIdx = j
        scores[i], scores[maxIdx] = scores[maxIdx], scores[i]


def countRankTot(predictor, scores):
    #predictor is either 'PWE' or 'IWE'
    tot = 0
    for i in range(len(scores)):
        if scores[i, 0] == predictor:
            tot += i+1
    return tot

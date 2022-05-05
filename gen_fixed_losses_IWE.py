#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#graph fixing. The IWE's losses have a floor that only about -0.194. This is
#because of the dead (black) normals in the GT normal maps. To correct for this
#and make comparisons with the PWE's losses, we should multiply IWE's losses by
#1/0.194 
import pickle
from grapher import graphTraining, graphTesting
import os
#import sys

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

with open('trainLoss_IWE.data', 'rb') as trainLossIn:
    trainLosses = pickle.load(trainLossIn)

with open('testLoss_IWE.data', 'rb') as testLossIn:
    testLosses = pickle.load(testLossIn)

steps = trainLosses[-1][0]

for k in range(len(trainLosses)):
    trainLosses[k][1] = trainLosses[k][1] * (1/0.194)

for k in range(len(testLosses)):
    testLosses[k][1] = testLosses[k][1] * (1/0.194)

fixedProgPath = os.path.join(fileDirectory, 'Fixed IWE Progress')

graphTraining(trainLosses, steps, fixedProgPath)
graphTesting(testLosses, steps, fixedProgPath)

print('done')

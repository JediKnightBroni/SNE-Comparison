#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#make colour scale for angular error. It will a spectrum of red colours

import numpy as np
from PIL import Image
import os
#import sys

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

grids = np.mgrid[0:180, 0:180]
redChannel = np.ones((180, 180, 1))
redChannel = np.multiply(redChannel, 180)
altChannel = np.reshape(grids[0], (180, 180, 1))
scale = np.concatenate((redChannel, altChannel, altChannel), axis=-1)

scale = np.floor(np.multiply(scale, 255/180))
img = Image.fromarray(np.uint8(scale), 'RGB')
imgFileName = 'colour_scale.png'
imgFilePath = os.path.join(fileDirectory, imgFileName)
img.save(imgFilePath)

print('done')

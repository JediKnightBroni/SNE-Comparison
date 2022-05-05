#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#generate GT Normal maps

#import numpy as np
import comparator_helper
import os
#import sys

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

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

D = 5
T = 2
DIM = 224

for d in range(1, D+1):
    for t in range(T):
        pair = comparator_helper.getFilePair(imgDirs, normDirs, dirIDRanges, d, t)
        comparator_helper.saveMapToImage(pair[1], pair[1], d, t, 'ground_truth')

print('done')

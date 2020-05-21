# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:20:40 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\Registration\\')
import MyFunctions
import RigidRegistration as RR

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


volumeroot = 'E:\\human\\'
trainlist = []
testlist = []
FrameNum = 5
radius = 7  

for file in os.listdir(volumeroot):
    if file.startswith('Retina2_ONH'):
        trainlist.append(file)
trainlist.sort()

for file in os.listdir(volumeroot):
    if file.startswith('Retina2_Fovea'):
        testlist.append(file)
testlist.sort()

#%%
reglist = testlist

for i in range(len(reglist)):
    filename = reglist[i]
    print('processing {}'.format(filename))
    
    volume = MyFunctions.ImageRescale(io.imread(volumeroot+filename),[0,255])
    volume = RR.VolumeReshape(volume,FrameNum)
    _,avers = RR.FrameRegistration(volume,FrameNum,True) 
    pair = RR.BscanRegistration(avers,radius,True)
    
    with open('E:\\VoxelMorph\\'+filename[:-4]+'.pickle','wb') as f:
        pickle.dump(pair,f)


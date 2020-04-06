# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:07:54 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import MotionCorrection
import MyFunctions

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

'''
1. Mice data reshape to [500,512,512]
2. 5-repeated frame, pick 1st as the reference
3. Do motion correction and save in form of packages in a tuple
'''
global FrameNum, radius
FrameNum = 5
radius = 7

Num_volume = 2

def PickFrame(volume):
    dim = volume.shape
    opt = np.zeros([int(dim[2]/FrameNum),512,512],dtype=np.float32)
    for i in range(dim[2]):
        if i % FrameNum == 0:
            opt[int(i/FrameNum),:,:] = volume[:512,:512,i]
    return opt

volumeroot = 'E:\\mouse\\'
xlist = []

for file in os.listdir(volumeroot):
    if file.endswith('structure.nii'):
        xlist.append(file)
xlist.sort()

#%%
temproot = 'E:\\Temp\\'
fixedImageFile = temproot+'fix_img.nii.gz'
movingImageFile = temproot+'mov_img.nii.gz'
outputImageFile = temproot+'opt.nii.gz'

t1 = time.time()

for vol in range(Num_volume):
    pair = ()
    # Use Num_volume to train the VoxelMorph
    data = np.uint8(PickFrame(MyFunctions.nii_loader(volumeroot+xlist[vol])))
    dim = data.shape
    
    for i in range(dim[0]):
        if i >= radius and i < dim[0]-radius:
            # first define a fix image as a center
            y = data[i,:,:]
            MyFunctions.nii_saver(y,temproot,'fix_img.nii.gz')
            # go through all B-scans in radius as moving image
            for j in range(radius):
                dist = j+1
                
                # frame before fixed image
                move = data[i-dist,:,:]
                MyFunctions.nii_saver(move,temproot,'mov_img.nii.gz')
                MotionCorrection.MotionCorrect(fixedImageFile,
                                        movingImageFile, outputImageFile)
                x_pre = np.zeros([512,512],dtype=np.float32)
                x_pre = MyFunctions.nii_loader(outputImageFile)
                
                # frame after fixed image
                move = data[i+dist,:,:]
                MyFunctions.nii_saver(move,temproot,'mov_img.nii.gz')
                MotionCorrection.MotionCorrect(fixedImageFile,
                                        movingImageFile, outputImageFile)
                x_post = np.zeros([512,512],dtype=np.float32)
                x_post = MyFunctions.nii_loader(outputImageFile)
                
                pair = pair + ((x_pre,y),(x_post,y),)
            pair = pair + ((y,y),)
    
    print('Volume {} has been paired'.format(vol))
    
    with open('E:\\VoxelMorph\\'+xlist[vol][:-4]+'.pickle','wb') as f:
        pickle.dump(pair,f)
    del pair

t2 = time.time()
print('Time:{} min'.format((t2-t1)/60))          
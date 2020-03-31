# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:48:54 2020

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
from skimage import io

'''
1.Human data need to be rescaled to [0,255]
2.Pick single frame 
3.Do motion correction and save in form of packages in a tuple 
'''
global FrameNum
FrameNum = 5

def PickFrame(volume):
    dim = volume.shape
    opt = np.zeros([int(dim[0]/FrameNum),dim[1],dim[2]],dtype=np.float32)
    for i in range(dim[0]):
        if i % FrameNum == 0:
            opt[int(i/FrameNum),:,:] = volume[i,:,:]
    return opt

volumeroot = 'E:\\human\\'
trainlist = []

for file in os.listdir(volumeroot):
    if file.startswith('Retina2') and file.endswith('1.tif'):
        trainlist.append(file)
trainlist.sort()

#%% Pack
'''
Go through all volumes in the training list/testing list
'''

radius = 7
root = 'E:\\Temp\\'
fixedImageFile = root+'fix_img.nii.gz'
movingImageFile = root+'mov_img.nii.gz'
outputImageFile = root+'opt.nii.gz'

t1 = time.time()

for vol in range(len(trainlist)):
    
    volume = MyFunctions.ImageRescale(io.imread(volumeroot+trainlist[vol]),
                                      [0,255])
    data = PickFrame(volume)
    del volume
    dim = data.shape
    pair = ()
    
    for i in range(dim[0]):
        if i >= radius and i < dim[0]-radius:
            fix = data[i,:,:] 
            MyFunctions.nii_saver(fix,root,'fix_img.nii.gz')
            for j in range(radius):
                dist = j+1
                frame_x = data[i-dist,:,:]
                MyFunctions.nii_saver(frame_x,root,'mov_img.nii.gz')
                MotionCorrection.MotionCorrect(fixedImageFile,movingImageFile,outputImageFile)
                x_pre = np.zeros([1024,512],dtype=np.float32)
                x_pre[:,:500] = MyFunctions.nii_loader(outputImageFile)
                
                frame_x = data[i+dist,:,:]
                MyFunctions.nii_saver(frame_x,root,'mov_img.nii.gz')
                MotionCorrection.MotionCorrect(fixedImageFile,movingImageFile,outputImageFile)
                x_post = np.zeros([1024,512],dtype=np.float32)
                x_post[:,:500] = MyFunctions.nii_loader(outputImageFile)
                
                y = np.zeros([1024,512],dtype=np.float32)
                y[:,:500] = fix
                pair = pair + ((x_pre,y),(x_post,y),)
            pair = pair + ((y,y),)
            
        if i % 20 == 0 :
            print('{} slices have been completed'.format(i))
            
    with open('E:\\VoxelMorph\\'+trainlist[vol][:-4]+'.pickle','wb') as f:
        pickle.dump(pair,f)
    del pair
    
t2 = time.time()
print('Time:{} min'.format((t2-t1)/60))


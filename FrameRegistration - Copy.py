# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:03:13 2020

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
1. Human data resacle to [0,255]
2. 5-repeated frame (1 is reference)
3. Motion correction and save in a tuple
'''

global FrameNum
FrameNum = 5

volumeroot = 'E:\\human\\'
trainlist = []

for file in os.listdir(volumeroot):
    if file.startswith('Retina2') and file.endswith('1.tif'):
        trainlist.append(file)    

trainlist.sort()

#%% Frame registration
temproot = 'E:\\Temp2\\'
fixedImageFile = temproot+'fix_img.nii.gz'
movingImageFile = temproot+'mov_img.nii.gz'
outputImageFile = temproot+'opt.nii.gz'

t1 = time.time()

for vol in range(len(trainlist)):
    # x-y pair saved in a tuple
    pair = ()
    # Train on Retina1_ONH
    ipt = io.imread(volumeroot+trainlist[vol])
    ipt = MyFunctions.ImageRescale(ipt,[0,255])
    dim = ipt.shape
    
    for slc in range(dim[0]):
        if slc % FrameNum == 0:
            frames = np.zeros([5,1024,512],dtype=np.float32)
            
            # Pick the first frame as reference
            fix = ipt[slc,:,:]
            MyFunctions.nii_saver(fix,temproot,'fix_img.nii.gz')
            frames[0,:,:500] = fix
    
            # iteratively register the following 4 frames        
            for i in range(1,5):
                mov = ipt[slc+i,:,:]
                MyFunctions.nii_saver(mov,temproot,'mov_img.nii.gz')
                MotionCorrection.MotionCorrect(fixedImageFile, movingImageFile,
                                               outputImageFile)
                frames[i,:,:500] = MyFunctions.nii_loader(outputImageFile)
            
            # 5-average and pair
            x = frames[0,:,:]
            y = np.mean(frames,axis=0)
            pair = pair + ((x,y),)
        
        if slc % 100 == 0 :
            print('Processing: [%d/%d]'%(slc,dim[0]))
            
    with open(volumeroot+trainlist[vol][:-4]+'_FR'+'.pickle','wb') as f:
        pickle.dump(pair,f)
    
    del pair
    
t2 = time.time()
print('Time: {} min'.format((t2-t1)/60))

#%% Visualization
#x_volume = np.zeros([len(pair),1024,512],dtype=np.float32) 
#y_volume = np.zeros([len(pair),1024,512],dtype=np.float32) 
#
#for i in range(len(pair)):
#    x_volume[i,:,:] = pair[i][0]
#    y_volume[i,:,:] = pair[i][1]
#
#MyFunctions.nii_saver(x_volume,temproot,'x_volume.nii.gz')
#MyFunctions.nii_saver(y_volume,temproot,'y_volume.nii.gz')

            
            
        









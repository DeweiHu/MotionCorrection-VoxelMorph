#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:56:01 2020

@author: dewei
"""

import sys
sys.path.insert(0,'/home/dewei/Desktop/Denoise/')
import MyFunctions

import os
import subprocess
import pickle
import numpy as np
import time
from skimage import io
from PIL import Image

root = '/home/dewei/Desktop/Registered_volume/'
temp = '/home/dewei/Desktop/slc/'
radius = 7

volumelist = []
for file in os.listdir(root):
    if file.endswith('.pickle'):
        volumelist.append(file)


#%%

t1 = time.time()

for vol in range(len(volumelist)):
    filename = volumelist[vol]
    with open(root+filename,'rb') as f:
        volume = pickle.load(f)
    
    volume_sf = np.zeros([len(volume),1024,512],dtype=np.float32)
    
    for i in range(len(volume)):
        # Go through all sclices, do self-fusion
        pack = np.squeeze(volume[i],axis=1)
        for j in range(2*radius+1):
            mov = Image.fromarray(pack[j,:,:])
            mov.save(temp+'atlas{}.tif'.format(j))
        fix = Image.fromarray(pack[-1,:,:])
        fix.save(temp+'fix_img.tif')
        subprocess.call("/home/dewei/self_fusion.sh")
        
        # load the self-fusion result and save in a volume
        volume_sf[i,:,:] = io.imread(temp+'synthResult.tif')
    
    MyFunctions.nii_saver(volume_sf,root,filename[:-7]+'SF.nii.gz')
    
    del volume_sf,volume
    
t2 = time.time()
print('time: {} min'.format((t2-t1)/60))


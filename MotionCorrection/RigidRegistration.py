# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:40:28 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import MotionCorrection

import time
import numpy as np

def VolumeReshape(Volume,FrameNum):
    [nfnd,nr,nc] = Volume.shape
    nd = int(nfnd/FrameNum)
    opt = np.zeros([FrameNum,nr,nc,nd],dtype=np.float32)
    for i in range(nfnd):
        slc = int(np.floor(i/FrameNum))
        frame = int(i % FrameNum)
        opt[frame,:,:,slc] = Volume[i,:,:]
    return opt

'''
FrameRegistration function
input: volume [nf,nr,nc,nd]
       frame number
       result volume root
'''

def FrameRegistration(Volume,FrameNum,verbose):
    [nf,nr,nc,nd] = Volume.shape
    bscans = np.zeros([nf,nr,nc,nd],dtype=np.float32)
    avers = np.zeros([nr,nc,nd],dtype=np.float32)
    
    t1 = time.time()
    
    for slc in range(nd):
        # use the 1st frame as the reference
        fix = np.ascontiguousarray(np.float32(Volume[0,:,:,slc]))
        for f in range(nf):
            mov = np.ascontiguousarray(np.float32(Volume[f,:,:,slc]))
            bscans[f,:,:,slc] = MotionCorrection.MotionCorrect(fix,mov)
        avers[:,:,slc] = np.mean(bscans[:,:,:,slc],axis=0)
    
    t2 = time.time()
    
    if verbose == True:
        print('Number of registration: %d,\t Time consumed: %.4f min'
              %(nd*nf,(t2-t1)/60))
    return bscans, avers

'''
BscanRegistration: 
    Rigid registration between neighboring bscans within radius
    The output will be a (x,y) pairs save in tuple that will be used in voxelmorph
    So, there is a padding included.
'''
def BscanRegistration(Volume,radius,verbose):
    [nr,nc,nd] = Volume.shape
    pair = ()
    
    t1 = time.time()
    for i in range(nd):
        if i >= radius and i < nd-radius:
            fix = np.ascontiguousarray(np.float32(Volume[:,:,i]))
            for j in range(radius):
                dist = j+1

                mov_pre = np.ascontiguousarray(np.float32(Volume[:,:,i-dist]))
                opt_pre = MotionCorrection.MotionCorrect(fix,mov_pre)
                
                mov_post = np.ascontiguousarray(np.float32(Volume[:,:,i+dist]))
                opt_post = MotionCorrection.MotionCorrect(fix,mov_post)
                
                pair = pair + ((opt_pre,fix),(opt_post,fix),)
            pair = pair + ((fix,fix),)
        
        if i % 20 == 0:
            print('[%d/%d] complete'%(i,nd))
    t2 = time.time()
    
    if verbose == True:
        print('Number of registration: %d,\t Time consumed: %.4f min'
              %((2*radius+1)*(nd-2*radius+2),(t2-t1)/60))
              
    return pair
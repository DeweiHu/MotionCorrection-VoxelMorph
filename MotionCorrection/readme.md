# Motion Correction
The motion correction refers to compensate the deviation in 2 directions which comes from the motion of patient
1. Between the repeated frames
2. Between consecutive bscans
The correction is done by rigid registration. In this application, I used the itk library. For the 1st dimension, I 
used the first frame as the reference image. While in the second dimension, the fixed image is the center one, and 
moving images are neighboring bscans within a certain radius (r=7).

## Functions
1. VolumeReshape
Re-arange the volume to [nf,nr,nc,nd].

2. FrameRegistration

3. BscanRegistration

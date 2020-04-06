# MotionCorrection-VoxelMorph
The motion correction -- VoxelMorph pipeline major serves for human OCT denoising. The motion correction is a itk-based rigid registration,
and the VoxelMorph is a learning based deformable registration.

## VoxelMorph files
1. VoxelMorph_Train.py
2. VoxerMorph_test.py
3. losses.py

## Motion Correction files
1. MotionCorrection.py
2. FrameRegistration.py 
3. VolumeMotionCorrection.py

## Other file
1. MyFunctions.py
2. RunBash.py
3. VMC_mice.py (Do volume motion correction, prepare mice data to train VoxelMorph)

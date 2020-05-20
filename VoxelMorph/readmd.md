# VoxelMorph
The VoxelMorph is a learning-based deformable registration method. Basically, the network takes the fixed and moving image
as a pair of input. The output is the deformation field which can be applied to the moving image to get the registration
result. The loss function is defined by the difference/similarity of the registered image and reference image.

# Triplet-Learning
Training code to reproduce FaceNet

Code for Learning the Triplet Based Network

Changes compared to OpenFace:
- add cudnn modules for Networks
- get idea of "margin" from Oxford (because of smaller batch and less data)
- speed up choosing triplets
- speed up learning ~3x by only doing one F/B pass of model (OpenFace have 4)


Results:
- on CASIA it get better results than original code, up to 88%. 
- using Felix_V1(CASIA+FaceScrub) with some noise produce ~91%


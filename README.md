# TwindomNet
create a good mask
=======
# Full body 3D scanners for 3D figurines, 3D portraits and 3D selfies
This project is to convert PortraitFCN+ from Matlab to Tensorflow (or something else FOSS), then feed the outputs from it (converted to a trimap) into the Deep Image Matting paper's code, supervised by Richard Berwick.

Second stage is to explore creating a phone-to-model pipeline to match a parameterized body model to iPhone photo (front, side) silhouettes and height/gender inputs so we have a relatively accurate body to use for fitting.

Please note you might need to request Xiaoyeng's paper's code for dataset in order to train and run certain files, which we use to reference for preprocessing purposes. You might also need to pull certain dataset from PASCOL. 

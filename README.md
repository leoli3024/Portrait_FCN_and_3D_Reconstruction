
# High-level Objective
This project is to convert PortraitFCN+ (by Xiaoyong Shen, found [here](http://xiaoyongshen.me/webpage_portrait/index.html])) from Matlab to Tensorflow, then refine the outputs from it (converted to a trimap) using KNN & ResNet, supervised by Richard Berwick.

Second stage is to explore creating a 3D face-model using only one 2D-image.

This project is presentated on May 2nd at Sutardja Center of Entrepreneurship & Technology (SCET). The presentation slide can be assessed [here](https://github.com/leoli3024/Portrait-FCN-and-3D-reconstruction/blob/master/Portrait-FCN-3D-reconstruction.pdf).

# Acknowledge
Thanks to Xiaoyong Shen and his team for making this project possible. Their paper introducing this algorithm - Portrait FCN+ - can be found [here](http://xiaoyongshen.me/webpage_portrait/papers/portrait_eg16.pdf). We also would like to appreciate Patrik Huber and his team for enabling 3D face reconstruction using basic and simple 2D-inputs. Find their the [paper](http://www.patrikhuber.ch/files/3DMM_Framework_VISAPP_2016.pdf) - A Multiresolution 3D Morphable Face Model and Fitting Framework, P. Huber, G. Hu, R. Tena, P. Mortazavian, W. Koppen, W. Christmas, M. Rätsch, J. Kittler - at International Conference on Computer Vision Theory and Applications (VISAPP) 2016. 

Please note you might need to request Xiaoyeng's paper's code for dataset in order to train and run certain files, which we use to reference for preprocessing purposes. You might also need to pull certain dataset from PASCOL. The training datasets could also be accessed here: https://drive.google.com/file/d/1TuVO2N_vthca4_B8GhT4JFbV2CJVVIW7/view?usp=sharing.

# Twindom - Deep Image Segmentation in Tensorflow (Portrait FCN+)

The green screen technique, used by filmmakers and producers in Hollywood, involves filming actors in front of a uniform background, which is typically green. This process allows editors to easily separate the subjects from the background and insert a new background, but it has two main problems: time and cost. In fact, a Hollywood movie can takes 6 months to post-produce/edit. Moreover, the most recent Avengers movie Avengers: Infinity War had a budget of $320 million with 25% of that going to production costs. Time and cost problems are surmountable for Hollywood movies but can be deal-breakers for would-be filmmakers. 

## Overview
Our solution automates the green screen & editing process. We take in any image of a person and output a state of the art alpha matte, an image of black (background) and white (foreground/person), which can be used to isolate foreground. Our algorithm is shown below:

- 1) Our preprocessing stage captures the positional offsets of all our portrait inputs with respect to a reference image using deep metric learning, a facial featurization technique. We a) identify 49 points which captures distinct facial features for both images b) find the affine transformation between these two sets of points and c) outputs the affine transforms of the mean mask & positional offsets of the reference. 

- 2) We take the 3 channels outputted via preprocessing and the portrait as inputs into a 20 layer encoder-decoder network called Portrait FCN+, which outputs an unrefined alpha matte. We train Portrait FCN+ on Amazon EC2 against the ground truth alpha matte (i.e. true subject area) of the images. We generate a trimap, an alpha matte with an additional region in grey representing unknown, by setting 10 pixels on either side of the subject and background segmentation line as the unknown.

- 3) The trimap is then put into two refinement stages: a)  KNN-matting applies K-nearest neighbors to classify the unknown (grey) region and b) ResNet deals with miniscule errors that might have occurred in the PortraitFCN+. The output here is an alpha matte. Our refinement algorithm is much less computationally expensive than the current state of the art refinement procedure, DIM, while maintaining the same accuracy: a 97% IoU. In fact, we shown that our refine-ment algorithm work on a Launchpad setup of 4KB, a miniscule amount compared to an IPhone, which has 64-256 GB.

![hardware setup](https://github.com/leoli3024/Portrait-FCN-and-3D-reconstruction/blob/master/Articles_Reports/Images/Hardware_setup.jpeg =640x640)

(Hardware setup testing computational advantage of KNN) 

![overview of solution](https://github.com/leoli3024/Portrait-FCN-and-3D-reconstruction/blob/master/Articles_Reports/Images/Overview.jpeg =640x640)
![training result](https://github.com/leoli3024/Portrait-FCN-and-3D-reconstruction/blob/master/Articles_Reports/Images/training_result.png =640x640)

## User Interface
We made a website built on Flask in which users can upload a portrait and receive a trimap, alpha matte, and a new image of themselves on a different background. The image is then run by our algorithm and then outputted onto either a web/mobile framework.  

![userinterface](https://github.com/leoli3024/Portrait-FCN-and-3D-reconstruction/blob/master/Articles_Reports/Images/User_interface.jpeg =640x640)

## Work in Progress - 3D Morphable Face Model	

We are currently on generating a 3D model from the segmented image. We use isomap texture extraction to obtain a pose-invariant representation of facial features and compute the linear-scaled orthographic projection camera pose estimation. We then fit these features onto the pre-developed Surrey Face Model, which is a Principal Component Analysis (PCA) model of shape variation built from 3D face scans. 

## Applications
Our project automates the process of separating the subject from the background in an image, having the potential to replace an entire sector of tasks in Hollywood, VR, and 3D-printing, all of which still uses the green screen techniques. In its current form, our project is useful for amateur photographers and filmmakers looking to change the background of an image, as we kept the whole runtime ~5 minutes, which is shorter than how long manually segmenting an image would take. Our project could be extended to segment video files, which would go a long way in automating the green screen technique. We are also currently working on creating a 3D model from the segmented image.






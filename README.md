# DSAC++ Code Documentation

- [General Setup](#general-setup)
- [Programs](#programs)
- [Data Structure](#data-structure)

## Introduction

This document explains the high level concepts and setup of our improved DSAC code for camera localization. See it in action [here](https://www.youtube.com/watch?v=DjJFRTFEUq0), and also see the [project page](https://hci.iwr.uni-heidelberg.de/vislearn/research/scene-understanding/pose-estimation/#CVPR18). You find an explanation of the method and theory in the following paper. Please cite this paper if you use this code in your own work:

E. Brachmann,  C. Rother,  
”[Learning Less is More - 6D Camera Localization via 3D Surface Regression](https://arxiv.org/abs/1711.10228)”,  
CVPR 2018

We publish the code under the BSD License 2.0. The predecessor of this pipeline can be found [here](https://github.com/cvlab-dresden/DSAC).

Compiling the code creates four programs:

* `train_obj`: Trains a scene coordinate regression CNN by optimizing the 3D distance between target scene coordinates and the CNN predictions (see Sec. 2.4. in our paper, "Scene Coordinate Initalization").
Target scene coordinates are either calculated from depth images (measured, rendered or estimated) or using our scene coordinate heuristic. 
The method requires RGB images, ground truth camera poses, and (optionally) depth images. 
* `train_repro`: Trains a scene coordinate regression CNN by optimizing the reprojection error of the CNN predictions (see Sec. 2.4. in our paper, "Optimization of Reprojection Error").
The method requires RGB images and ground truth camera poses, and a CNN model initialized e.g. by `train_obj`.
* `train_ransac`: Trains the whole camera localization pipeline end-to-end using DSAC (see Sec. 2.4. in our paper, "End-to-End Optimization.").
The method requires RGB images and ground truth camera poses, and a CNN model.
* `test_ransac`: Runs the camera localization pipeline on test images.
The method requires RGB images and a CNN model. 
Because the method calculates evaluation metrics, ground truth poses should also be provided.

Compiling the code will require: OpenCV 2.4, PNG++, Torch and cuDNN. We compiled the code using Ubuntu 16.04 and GCC 4.9.1.

In the following, we will first give more details about the general setup and the individual programs. 
Then, we describe how to run the code on the datasets used in the paper.
In case you have questions regarding the code, feel free to contact the first author of the associated paper.

## General Setup

The DSAC++ source code is a combination of a C++ framework and multiple Torch scripts. 
All data access, geometric operations and accuracy evaluation is performed in C++. 
The Torch scripts are called from C++ to grant access to the CNNs used in the pose estimation pipeline. 
We use two separate Torch scripts, one for regressing scene coordinates given an RGB image, and another one regressing hypothesis scores given re-projection error images. 
Note that the second script does not load a learnable CNN but instead constructs a soft inlier count for easy back propagation and the implementation of our entropy control schema (see Sec. 2.3. of our paper).
The Torch scripts are contained in the sub-folder core/lua. The script variants ending with nomodel are used when training our pipeline with the scene coordinate heuristic. 
They offer the exact same functionality as the standard scripts but store their respective CNNs under a different file name.

### Setting Parameters

All C++ programs share a set of parameters which the user can specify, see below for a complete list. 
There are two ways of setting the parameters. The first and direct way is to append it to the program call in the command line. 
Every parameter has a few letter abbreviation (case sensitive, see below) which can be given to the command line starting with a dash and followed by a space and the value to set, e.g. -ih 480 -iw 640. 
Because there might be a large number of parameters to set, there is also the second way which is providing a default.config file. 
This file should contain one parameter per line. 
More specifically, each line should start with the parameter abbreviation (without a leading dash) followed by a space and the value to set. 
The config file can also include comment lines which start with the # symbol. 
All programs will first look for the default.config file which will be parsed at the very beginning of each run. 
After that, command line parameters will be processed (overwriting parameters from the default.config). 
Our data package (see below) contains an example default.config file for the 3 datasets used in the experiments of the paper.

**Parameters related to the data:**

**iw** width of input images (px)

**ih** height of input images (px)

**fl** focal length of the RGB camera

**xs** x position of the principal point of the RGB camera

**ys** y position of the principal point of the RGB camera

**oscript** Torch script file for learning scene coordinate regression

**sscript** Torch script file for hypothesis scoring

**omodel** file storing the scene coordinate regression CNN

**cd** constant depth prior in meters to be used for the scene coordinate heuristic (set 0 to use provided depth images)

**sid** arbitrary string to be attached to output files (but does not affect CNN model file names), handy to differentiate mutliple runs of the pipeline

**iSS** sub-sampling of training images, e.g. iSS 10 will use 10% of the training data

**cSS** sub-sampling of the CNN architecture output w.r.t. the input (necessary for data exchange between C++ and Torch, e.g. the paper architecture takes an 640x480 image and predicts 80x60 coordinates, i.e. cSS is 8)

**Parameters related to pose estimation:**

**rdraw** draw a hypothesis randomly (rdraw 1), i.e. DSAC, or take the one with the largest score (rdraw 0), i.e. RANSAC

**rT** re-projection error threshold (in px) for measuring inliers in the pose estimation pipeline

**rI** initial number of pose hypotheses drawn per frame

**rRI** number of refinement iterations

**Parameters related to learning:**

Parameters which concern the learning of the scene coordinate regression CNN and the soft inlier score (e.g. learning
rate or file names to store the CNNs) are defined directly in the corresponding Torch scripts (see core/lua).
There are also some parameters defined in the respective main cpp file of each program, such as the total number of training iterations. 

### Training Procedure

As stated in the paper, end-to-end training (i.e. calling `train_ransac`) needs a good initialization in terms of the scene coordinate regression CNN. 
Therefore, pre-training (i.e. calling `train_obj`, potentially followed by `train_repro`) should be executed
first. 
An example order of executing the programs is given at further below.

### Scene Coordinate Ground Truth

In order to pre-train the scene coordinate regression CNN (i.e. calling `train_obj`), scene coordinate ground truth has to be available for each training frame. 
Our code generates scene coordinate ground truth from depth frames, the ground truth poses and camera calibration parameters. 
Depth frames can come from a RGB-D camera, a renderer or an arbitrary depth estimation algorithm.
Alignment is assumed between RGB and depth images.
Alternatively, `train_obj` can utilize a scene coordiante heuristic based on a constant depth assumption (parameter **cd** in meters). 
In this case, the scene coordinate regression CNN should be further optimized using `train_repro` for good accuracy.

## Programs

In the following, we describe input and output of the individual programs created by compiling the code. 
An example order of executing the programs is given at the end of the document.

### `train_obj`

The program `train_obj` pre-trains a scene coordinate regression CNN by minimizing the L1 distance between the predicted coordinate and the ground truth coordinate. 
The program needs access to a Torch script which constructs and grants access to a scene coordinate regression CNN (default: `train obj.lua`).
The program will store the current state of the CNN in a fixed interval of parameter updates. 
This interval and the file name is defined in the Torch script.
The program creates a training loss txt file which contains per line the training iteration number followed by the training loss at that iteration (mean over an image). 
When no depth images are available, the **cd** parameter should be set to a positive value in meters (e.g. 3 for indoor and 10 for outdoor scenes).
The program will then calculate ground truth scene coordinate assuming a constant depth for each image.

### `train_repro`

The program `train_repro` loads a pre-trained scene coordinate regression CNN and refines it by optimizing the reprojection error of its predictions. 
The program needs access to a Torch script to load and access the scene coordinate regression CNN (default: `train_obj.lua`).
The CNN to load can be specified via -omodel. 
The program will store the current states of both CNNs in fixed intervals of parameter updates. 
These intervals and the file names are defined in the Torch scripts. 
The program also creates a repro training loss txt file which contains the training loss per training
iteration. 

### `train_ransac`

The program `train_ransac` loads a pre-trained scene coordinate regression CNN and refines it by training the complete camera localization pipeline end-to-end using the DSAC strategy. 
The program optimizes the expectation of the pose loss. 
The program needs access to two Torch scripts to load and access the scene coordinate regression CNN (default: `train_obj_e2e.lua`) and the soft inlier score (default: `score_incount_ec6.lua`). 
The CNN to load can be specified via -omodel. 
The program will store the current states of both CNNs in fixed intervals of parameter updates. 
These intervals and the file names are defined in the Torch scripts. 
The program also creates a ransac training loss txt file which contains training statistics per training
iteration (see `train_ransac.cpp`, bottom for a listing). 

### `test_ransac`

The program test ransac loads a scene coordinate regression CNN and performs camera localization on a set of test images using
RANSAC (-rdraw 0) or DSAC -rdraw 1. 
The program needs access to two Torch scripts to load and access the scene coordinate regression CNN (default: `train_obj.lua`) and the soft inlier score (default: `score_incount_ec6.lua`). 
The CNNs to load can be specified via -omodel. 
The program creates a ransac test loss txt file which contains statistics of the test run (see `test_ransac.cpp` bottom for a listing). 
Most importantly, the first number is the ratio of test images with correctly estimated poses (i.e. a pose error below 5cm 5deg). Furthermore, a ransac test errors txt file will be created, which contains per line statistics for each test image, most importantly the estimated pose (see `test_ransac.cpp`, bottom for a complete listing).

## Data Structure

We provide a [data package](https://cloudstore.zih.tu-dresden.de/index.php/s/5hHv4gywLFI9K07) for deployment of our code for the datasets used in the paper. 
Note that we do not provide the data set itself, but merely the structure outline and associated meta data to reproduce the results
from our paper. (As an exception we provide rendered depth images for [7scenes](https://cloudstore.zih.tu-dresden.de/index.php/s/IS7AsAUioKKX7JZ), [cambridge](https://cloudstore.zih.tu-dresden.de/index.php/s/i90GR0smcjowGqT)). 
Our code assumes the following data structure:

1. `dataset`
2. `dataset\scene`
3. `dataset\scene\training`
4. `dataset\scene\training\depth`
5. `dataset\scene\training\rgb`
6. `dataset\scene\training\poses`
7. `dataset\scene\test`
8. `dataset\scene\test\depth`
9. `dataset\scene\test\rgb`
10. `dataset\scene\test\poses`
11. `dataset\scene\default.config`
12. `dataset\scene\Entropy.lua`
13. `dataset\scene\MyL1Criterion.lua`
14. `dataset\scene\score_incount_ec6.lua`
15. `dataset\scene\train_obj.lua`
16. `dataset\scene\train_obj_e2e.lua`
17. `dataset\scene\train_obj_nomodel.lua`
18. `dataset\scene\train_obj_nomodel.lua`

For example `dataset` could be 7Scenes and `scene` could be Chess. 
Folders 3-10 should be filled with the associated files from the data set, i.e. RGB frames, depth frames and pose files (7Scenes convention) according to the training/test split. 
For RGB images we support 8-bit 3-channel PNG or JPG files.
For depth images we support 16-bit 1-channel PNG files or TIFF files. 
The depth should be stored in millimeters. 
Config file 11 is specific to each dataset.
Files 11 to 18 are implemented as soft links in our data package. The soft links assume that the `dataset` lie within `core/build` (for access to `core/lua`.
The data package also contains fully trained models for each scene (after end-to-end refinement).

You find the rendered depth images we used for training with a 3D model here: [7scenes](https://cloudstore.zih.tu-dresden.de/index.php/s/IS7AsAUioKKX7JZ), [cambridge](https://cloudstore.zih.tu-dresden.de/index.php/s/i90GR0smcjowGqT)

### Executing the code

The following calls assume you are within `core/build/dataset/scene` and the binaries lie in `core/build`.

*When training with depth images:*

1. `../../train_obj -oscript train_obj.lua`
2. `../../train_repro -oscript train_obj.lua -omodel obj_model_fcn_init.net`
3. `../../train_ransac -oscript train_obj_e2e.lua -omodel obj_model_fcn_repro.net -sscript score_incount_ec6.lua`
4. `../../test_ransac -oscript train_obj.lua -omodel obj_model_fcn_e2e.net -sscript score_incount_ec6.lua -rdraw 0`

*When training without depth images:*

1. `../../train_obj -oscript train_obj_nomodel.lua -iSS 20 -cd 3` (for outdor scenes we used `-cd 10`)
2. `../../train_repro -oscript train_obj_nomodel.lua -omodel obj_model_fcn_init_nomodel.net`
3. `../../train_ransac -oscript train_obj_e2e_nomodel.lua -omodel obj_model_fcn_repro_nomodel.net -sscript score_incount_ec6.lua`
4. `../../test_ransac -oscript train_obj.lua -omodel obj_model_fcn_e2e_nomodel.net -sscript score_incount_ec6.lua -rdraw 0`

Note that you can test the pipeline after each training step (1-3) by providing the appropriate model file to `test_ransac`.

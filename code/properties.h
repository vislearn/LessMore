/*
Copyright (c) 2016, TU Dresden
Copyright (c) 2017, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#pragma once

#include "types.h"

#include <string>
#include <vector>

/** Here global parameters are defined which are made available throughout the code. */

/**
 * @brief Parameters that affect the pose estimation.
 */
struct TestingParameters
{
    std::string sessionString; // arbitrary string to be appended to output files

    std::string objScript; // lua script for learning object coordinate regression
    std::string scoreScript; // lua script for hypothesis score regression
    std::string objModel; // file storing the object coordinate regression CNN

    int ransacIterations; // initial number of pose hypotheses drawn per frame
    int ransacRefinementIterations; // number of refinement iterations
    float ransacInlierThreshold; // reprojection error threshold (in px) for measuring inliers in the pose pipeline

    bool randomDraw; // draw a hypothesis randomly (true) or take the one with the largest score (false)
};

/**
 * @brief Parameters that affect the data.
 */
struct DatasetParameters
{ 
    std::string config; // name of the config file to read (a file that lists parameter values)

    //dataset parameters
    float focalLength; // focal length of the RGB camera
    float xShift; // x position of the principal point of the RGB camera
    float yShift; // y position of the principal point of the RGB camera
    int imgPadding;
    
    int imageWidth; // width of the input images (px)
    int imageHeight; // height of the input images (px)
    
    float constD; // if positive value that a constant depth heursitic will be used when initializing scene coordinates
    int imageSubSample; // use only every n th dataset image

    int cnnSubSample; // sub sampling of the CNN output wrt the input

};

/**
 * @brief Singelton class for providing parameter setting globally throughout the code.
 */
class GlobalProperties
{
protected:
    /**
     * @brief Consgtructor. Sets default values for all parameters.
     */
    GlobalProperties();
public:
    // Dataset parameters
    DatasetParameters dP;
  
    // Testing parameters
    TestingParameters tP;

    /**
     * @brief Get a pointer to the singleton. It will create an instance if none exists yet.
     *
     * @return GlobalProperties* Singleton pointer.
     */
    static GlobalProperties* getInstance();

    /**
     * @brief Returns the 3x3 camera matrix consisting of the intrinsic camera parameters.
     *
     * @return cv::Mat_< float > Camera/calibration matrix.
     */
    cv::Mat_<float> getCamMat();

    /**
     * @brief Parse the arguments given in the command line and set parameters accordingly
     *
     * @param argc Number of parameters.
     * @param argv Array of parameters.
     * @return void
     */
    void parseCmdLine(int argc, const char* argv[]);

    /**
     * @brief Parse a config file (given by the global parameter "config") and set parameters accordingly
     *
     * @return void
     */
    void parseConfig();
    
    /**
     * @brief Process a list of arguments and set parameters accordingly.
     *
     * Each parameter consists of a dash followed by a abbreviation of the parameter. In most cases the next entry in the list should be the value that the parameter should take.
     *
     * @param argv List of parameters and parameter values.
     * @return bool
     */
    bool readArguments(std::vector<std::string> argv);
    
    /**
     * @brief Returns the image X dimension as passed to the CNN (without padding).
     * @return CNN input width.
     */
    int getCNNInputDimX(){return dP.imageWidth - 2 * dP.imgPadding;}

    /**
     * @brief Returns the image Y dimension as passed to the CNN (without padding).
     * @return CNN input height.
     */
    int getCNNInputDimY(){return dP.imageHeight - 2 * dP.imgPadding;}

    /**
     * @brief Returns the X dimension of the CNN output.
     * @return CNN output width.
     */
    int getCNNOutputDimX(){return ceil(getCNNInputDimX() / (float) dP.cnnSubSample);}

    /**
     * @brief Returns the Y dimension of the CNN output.
     * @return CNN output height.
     */
    int getCNNOutputDimY(){return ceil(getCNNInputDimY() / (float) dP.cnnSubSample);}

private:
    static GlobalProperties* instance; // singelton instance
};

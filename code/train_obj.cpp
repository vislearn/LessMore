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


#include <fstream>

#include "properties.h"
#include "thread_rand.h"
#include "util.h"
#include "stop_watch.h"
#include "dataset.h"

#include "lua_calls.h"
#include "cnn.h"

/**
 * @brief Transforms a coordinate image to a floating point CNN output format.
 *
 * The image will be subsampled according to the CNN output dimensions.
 *
 * @param obj Coordinate image.
 * @param sampling Subsampling information of CNN output wrt to RGB input.
 * @return Coordinate image in CNN output format.
 */
cv::Mat_<cv::Vec3f> getObjMap(const jp::img_coord_t& obj, const cv::Mat_<cv::Point2i>& sampling)
{
    cv::Mat_<cv::Vec3f> objMap(sampling.size());

    for(unsigned x = 0; x < sampling.cols; x++)
    for(unsigned y = 0; y < sampling.rows; y++)
    {
        objMap(y, x) = obj(sampling(y, x).y, sampling(y, x).x);
    }

    return objMap;
}

/**
 * @brief Calls the torch training function (forward + backward pass)
 * @param img Input RGB image.
 * @param labels Target output coordinates.
 * @param state LUA state.
 * @return Loss of current iteration.
 */
double train(const cv::Mat_<cv::Vec3f>& img, const cv::Mat_<cv::Vec3f>& labels, lua_State* state)
{
    std::vector<cv::Mat_<cv::Vec3f>> imgV;
    imgV.push_back(img);

    std::vector<cv::Mat_<cv::Vec3f>> labelV;
    labelV.push_back(labels);

    lua_getglobal(state, "train");
    pushMaps(imgV, state);
    pushMaps(labelV, state);
    lua_pcall(state, 2, 1, 0);
    
    double loss = lua_tonumber(state, -1);
    lua_pop(state, 1);
    
    return loss;
}

int main(int argc, const char* argv[])
{
    int trainingLimit = 300000; // total number of updates
    
    // read parameters
    GlobalProperties* gp = GlobalProperties::getInstance();
    gp->parseConfig();
    gp->parseCmdLine(argc, argv);
   
    std::string baseScriptRGB = gp->tP.objScript;

    std::cout << std::endl << BLUETEXT("Loading training set ...") << std::endl;
    jp::Dataset trainDataset = jp::Dataset("./training/");

    std::cout << "Found " << trainDataset.size() << " training images." << std::endl;

    // lua and model setup
    lua_State* state = luaL_newstate();
    luaL_openlibs(state);

    execute(baseScriptRGB.c_str(), state);
    constructModel(gp->getCNNInputDimX(), gp->getCNNInputDimY(), gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), state);
    setTraining(state);
        
    std::cout << GREENTEXT("Training CNN.") << std::endl;
    
    std::ofstream trainFile;
    trainFile.open("training_loss_"+baseScriptRGB+"_"+gp->tP.sessionString+".txt"); // contains the training loss per iteration

    cv::Mat_<cv::Point2i> sampling;
    StopWatch stopW;

    for(unsigned r = 0; r < trainingLimit; r++)
    {
        std::cout << BLUETEXT("Starting training round " << r) << std::endl;
        int imgIdx = irand(0, trainDataset.size());

        // load training image
        jp::img_bgr_t img;
        trainDataset.getBGR(imgIdx, img);

        jp::img_coord_t obj;
        trainDataset.getObj(imgIdx, obj);

        // convert to CNN format
        cv::Mat_<cv::Vec3f> imgMap = getImgMap(img, sampling, true);
        cv::Mat_<cv::Vec3f> objMap = getObjMap(obj, sampling);

        // pass to LUA for forward + backward pass
        float loss = train(imgMap, objMap, state);

        trainFile << r << " " << loss << std::endl;
        std::cout << YELLOWTEXT("Training loss: " << loss << " in " << stopW.stop() << "ms.")  << std::endl;
    }
    
    trainFile.close();

    lua_close(state);
    return 0;
}

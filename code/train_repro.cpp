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


#include <iostream>
#include <fstream>

#include "properties.h"
#include "thread_rand.h"
#include "util.h"
#include "stop_watch.h"
#include "dataset.h"

#include "lua_calls.h"
#include "cnn.h"

int main(int argc, const char* argv[])
{
    // read parameters
    GlobalProperties* gp = GlobalProperties::getInstance();
    gp->parseConfig();
    gp->parseCmdLine(argc, argv);

    int trainingRounds = 300000; // total number of updates

    std::string baseScriptRGB = gp->tP.objScript;
    std::string modelFileRGB = gp->tP.objModel; // initialized CNN

    std::cout << std::endl << BLUETEXT("Loading training set ...") << std::endl;
    jp::Dataset trainingDataset = jp::Dataset("./training/");

    // setup LUA and load model
    std::cout << "Loading script: " << baseScriptRGB << std::endl;
    lua_State* stateRGB = luaL_newstate();
    luaL_openlibs(stateRGB);
    execute(baseScriptRGB.c_str(), stateRGB);
    loadModel(modelFileRGB, gp->getCNNInputDimX(), gp->getCNNInputDimY(), gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), stateRGB);
    setTraining(stateRGB);

    cv::Mat camMat = gp->getCamMat();

    std::ofstream trainFile;
    trainFile.open("repro_training_loss"+gp->tP.objScript+"_"+gp->tP.sessionString+".txt"); // contains training loss per iteration

    for(unsigned round = 0; round <= trainingRounds; round++)
    {
        std::cout << YELLOWTEXT("Round " << round << " of " << trainingRounds << ".") << std::endl;

        int imgID = irand(0, trainingDataset.size());

        // load training image
        jp::img_bgr_t imgBGR;
        trainingDataset.getBGR(imgID, imgBGR);

        jp::cv_trans_t hypGT;
        trainingDataset.getPose(imgID, hypGT);

        // forward pass
        std::cout << BLUETEXT("Predicting object coordinates.") << std::endl;

        cv::Mat_<cv::Point2i> sampling;
        std::vector<cv::Mat_<cv::Vec3f>> imgMaps;
        jp::img_coord_t estObj = getCoordImg(imgBGR, sampling, imgMaps, true, stateRGB);

        std::cout << BLUETEXT("Calculating gradients wrt projection error.") << std::endl;
        StopWatch stopW;

        cv::Mat_<double> dLoss_dObj = cv::Mat_<double>::zeros(sampling.rows * sampling.cols, 3); // acumulate hypotheses gradients for patches
        double loss = 0;

        #pragma omp parallel for
        for(unsigned x = 0; x < sampling.cols; x++)
        for(unsigned y = 0; y < sampling.rows; y++)
        {
            cv::Point2f imgPt = sampling(y, x);
            cv::Point3f objPt(estObj(y, x));

            loss += project(imgPt, objPt, hypGT, camMat);
            cv::Mat_<double> dNdO = dProjectdObj(imgPt, objPt, hypGT, camMat);

            int dIdx = y * sampling.cols + x;
            dNdO.copyTo(dLoss_dObj.row(dIdx));
        }

        loss /= sampling.cols * sampling.rows;

        std::cout << GREENTEXT("Projection Loss: " << loss) << std::endl;

        // learning hyperparameters were tuned for object coordinates in mm
        // we rescale gradients here instead of scaling hyperparameters (learning rate, clamping etc)
        dLoss_dObj /= 1000.0;

        std::cout << BLUETEXT("Gradient statistics:") << std::endl;
        std::cout << "Max gradient: " << getMax(dLoss_dObj) << std::endl;
        std::cout << "Avg gradient: " << getAvg(dLoss_dObj) << std::endl;
        std::cout << "Med gradient: " << getMed(dLoss_dObj) << std::endl << std::endl;

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        // backward pass
        backward(loss, imgMaps, dLoss_dObj, stateRGB);

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
        stopW.stop();

        trainFile << round << " " << loss << std::endl;
        std::cout << std::endl;
    }

    trainFile.close();
    lua_close(stateRGB);

    return 0;
}

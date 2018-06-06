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

    int trainingRounds = 50000;
    
    int refSteps = gp->tP.ransacRefinementIterations;
    int objHyps = gp->tP.ransacIterations;
    int inlierThreshold2D = gp->tP.ransacInlierThreshold;
  
    std::string baseScriptRGB = gp->tP.objScript;
    std::string baseScriptObj = gp->tP.scoreScript;
    std::string modelFileRGB = gp->tP.objModel;
   
    std::cout << std::endl << BLUETEXT("Loading training set ...") << std::endl;
    jp::Dataset trainingDataset = jp::Dataset("./training/");
        
    // lua and models
    std::cout << "Loading script: " << baseScriptObj << std::endl;    
    lua_State* stateObj = luaL_newstate();
    luaL_openlibs(stateObj);
    execute(baseScriptObj.c_str(), stateObj);
    loadScore(inlierThreshold2D, gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), stateObj);
    setTraining(stateObj);
    
    std::cout << "Loading script: " << baseScriptRGB << std::endl;    
    lua_State* stateRGB = luaL_newstate();
    luaL_openlibs(stateRGB);
    execute(baseScriptRGB.c_str(), stateRGB);    
    loadModel(modelFileRGB, gp->getCNNInputDimX(), gp->getCNNInputDimY(), gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), stateRGB);
    setTraining(stateRGB);
       
    cv::Mat camMat = gp->getCamMat();
    
    std::ofstream trainFile;
    trainFile.open("ransac_training_loss"+gp->tP.objScript+"_"+gp->tP.sessionString+".txt"); // contains training information per iteration

    for(unsigned round = 0; round <= trainingRounds; round++)
    {
        std::cout << YELLOWTEXT("Round " << round << " of " << trainingRounds << ".") << std::endl;

        int imgID = irand(0, trainingDataset.size());

        // load training image
        jp::img_bgr_t imgBGR;
        trainingDataset.getBGR(imgID, imgBGR);

        jp::cv_trans_t hypGT;
        trainingDataset.getPose(imgID, hypGT);

        std::cout << BLUETEXT("Predicting object coordinates.") << std::endl;

        // forward pass
        cv::Mat_<cv::Point2i> sampling;
        std::vector<cv::Mat_<cv::Vec3f>> imgMaps;
        jp::img_coord_t estObj = getCoordImg(imgBGR, sampling, imgMaps, true, stateRGB);

        cv::Mat_<double> dLoss_dObj = cv::Mat_<double>::zeros(sampling.rows * sampling.cols, 3); // acumulate hypotheses gradients for patches

        StopWatch stopW;
        StopWatch globalStopW;

        double expectedLoss;
        double sfEntropy;
        bool correct;

        std::vector<jp::cv_trans_t> refHyps;
        std::vector<double> sfScores;
        std::vector<std::vector<cv::Point2i>> sampledPoints;
        std::vector<double> losses;
        std::vector<cv::Mat_<int>> inlierMaps;
        double tErr;
        double rotErr;
        int hypIdx;

        processImage(
            hypGT,
            stateObj,
            objHyps,
            camMat,
            inlierThreshold2D,
            refSteps,
            expectedLoss,
            sfEntropy,
            correct,
            refHyps,
            sfScores,
            estObj,
            sampling,
            sampledPoints,
            losses,
            inlierMaps,
            tErr,
            rotErr,
            hypIdx);

        // === doing the backward pass ====================================================================

        // --- path I, hypothesis path --------------------------------------------------------------------
        std::cout << BLUETEXT("Calculating gradients wrt hypotheses.") << std::endl;

        // precalculate gradients per of hypotheis wrt object coordinates
        std::vector<cv::Mat_<double>> dHyp_dObjs(refHyps.size());

        #pragma omp parallel for
        for(unsigned h = 0; h < refHyps.size(); h++)
        {
            // differentiate refinement around optimum found in last optimization iteration
            dHyp_dObjs[h] = cv::Mat_<double>::zeros(6, sampling.rows * sampling.cols * 3);

            if(sfScores[h] < EPS) continue; // skip hypothesis with no impact on expectation

            // collect inlier correspondences of last refinemen iteration
            std::vector<cv::Point2f> imgPts;
            std::vector<cv::Point2i> srcPts;
            std::vector<cv::Point3f> objPts;

            for(unsigned x = 0; x < inlierMaps[h].cols; x++)
            for(unsigned y = 0; y < inlierMaps[h].rows; y++)
            {
                if(inlierMaps[h](y, x))
                {
                    imgPts.push_back(sampling(y, x));
                    srcPts.push_back(cv::Point2i(x, y));
                    objPts.push_back(cv::Point3f(estObj(y, x)));
                }
            }

            if(imgPts.empty())
                continue;

            // calculate reprojection errors
            std::vector<cv::Point2f> projections;
            cv::Mat_<double> projectionsJ;
            cv::projectPoints(objPts, refHyps[h].first, refHyps[h].second, camMat, cv::Mat(), projections, projectionsJ);

            projectionsJ = projectionsJ.colRange(0, 6);

            //assemble the jacobean of the refinement residuals
            cv::Mat_<double> jacobeanR = cv::Mat_<double> ::zeros(objPts.size(), 6);
            cv::Mat_<double> dNdP(1, 2);
            cv::Mat_<double> dNdH(1, 6);

            for(int ptIdx = 0; ptIdx < objPts.size(); ptIdx++)
            {
                double err = std::max(cv::norm(projections[ptIdx] - imgPts[ptIdx]), EPS);
                if(err > CNN_OBJ_MAXINPUT)
                    continue;

                // derivative of norm
                dNdP(0, 0) = 1 / err * (projections[ptIdx].x - imgPts[ptIdx].x);
                dNdP(0, 1) = 1 / err * (projections[ptIdx].y - imgPts[ptIdx].y);

                dNdH = dNdP * projectionsJ.rowRange(2 * ptIdx, 2 * ptIdx + 2);
                dNdH.copyTo(jacobeanR.row(ptIdx));
            }

            //calculate the pseudo inverse
            jacobeanR = - (jacobeanR.t() * jacobeanR).inv() * jacobeanR.t();

            for(int ptIdx = 0; ptIdx < objPts.size(); ptIdx++)
            {
                cv::Mat_<double> dNdO = dProjectdObj(imgPts[ptIdx], objPts[ptIdx], refHyps[h], camMat);
                dNdO = jacobeanR.col(ptIdx) * dNdO;

                int dIdx = srcPts[ptIdx].y * sampling.cols * 3 + srcPts[ptIdx].x * 3;
                dNdO.copyTo(dHyp_dObjs[h].colRange(dIdx, dIdx + 3));
            }
        }

        // combine gradients per hypothesis
        std::vector<cv::Mat_<double>> gradients(refHyps.size());

        #pragma omp parallel for
        for(unsigned h = 0; h < refHyps.size(); h++)
        {
            cv::Mat_<double> dLoss_dHyp = dLossMax(refHyps[h], hypGT);
            gradients[h] = dLoss_dHyp * dHyp_dObjs[h];
        }

        for(unsigned h = 0; h < refHyps.size(); h++)
        for(unsigned idx = 0; idx < sampling.rows * sampling.cols; idx++)
        {
            dLoss_dObj(idx, 0) += sfScores[h] * gradients[h](idx * 3 + 0);
            dLoss_dObj(idx, 1) += sfScores[h] * gradients[h](idx * 3 + 1);
            dLoss_dObj(idx, 2) += sfScores[h] * gradients[h](idx * 3 + 2);
        }

        std::cout << BLUETEXT("Coord statistics:") << std::endl;
        std::cout << "Max gradient: " << getMax(dLoss_dObj) << std::endl;
        std::cout << "Avg gradient: " << getAvg(dLoss_dObj) << std::endl;
        std::cout << "Med gradient: " << getMed(dLoss_dObj) << std::endl << std::endl;

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        // --- path II, score path --------------------------------------------------------------------
        std::cout << BLUETEXT("Calculating gradients wrt scores.") << std::endl;

        std::vector<cv::Mat_<double>> dLoss_dScore_dObjs = dSMScore(estObj, sampling, sampledPoints, losses, sfScores, stateObj);

        // accumulate score gradients
        cv::Mat_<double> dLoss_dScore_dObj = cv::Mat_<double>::zeros(sampling.rows * sampling.cols, 3);

        for(unsigned h = 0; h < refHyps.size(); h++)
        {
            dLoss_dScore_dObj += dLoss_dScore_dObjs[h];
        }

        std::cout << BLUETEXT("Score statistics:") << std::endl;
        std::cout << "Max gradient: " << getMax(dLoss_dScore_dObj) << std::endl;
        std::cout << "Avg gradient: " << getAvg(dLoss_dScore_dObj) << std::endl;
        std::cout << "Med gradient: " << getMed(dLoss_dScore_dObj) << std::endl << std::endl;

        dLoss_dObj += dLoss_dScore_dObj;

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        std::cout << "Time of RANSAC Iteration: " << globalStopW.stop() / 1000 << "s." << std::endl;

        std::cout << BLUETEXT("Update object coordinate CNN.") << std::endl;

        // learning hyperparameters were tuned for object coordinates in mm
        // we rescale gradients here instead of scaling hyperparameters (learning rate, clamping etc)
        dLoss_dObj /= 1000.0;

        std::cout << BLUETEXT("Combined statistics:") << std::endl;
        int zeroGrads = 0;
        for(int row = 0; row < dLoss_dObj.rows; row++)
        {
            if(cv::norm(dLoss_dObj.row(row)) < EPS)
            zeroGrads++;
        }

        std::cout << "Max gradient: " << getMax(dLoss_dObj) << std::endl;
        std::cout << "Avg gradient: " << getAvg(dLoss_dObj) << std::endl;
        std::cout << "Med gradient: " << getMed(dLoss_dObj) << std::endl;
        std::cout << "Zero gradients: " << zeroGrads << std::endl;

        // backward pass
        backward(expectedLoss, imgMaps, dLoss_dObj, stateRGB);

        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
        globalStopW.stop();

        trainFile
            << round << " "              // 0 - iteration number
            << expectedLoss << " "       // 1 - expected loss
            << sfEntropy                 // 2 - entropy of score distribution
            << std::endl;

        std::cout << std::endl;
    }
    
    trainFile.close();
    
    lua_close(stateRGB);
    lua_close(stateObj);
    
    return 0;    
}

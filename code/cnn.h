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

﻿#pragma once

#define CNN_OBJ_MAXINPUT 100.0 // reprojection errors are clamped at this magnitude

#include "util.h"
#include "maxloss.h"

/**
* @brief Checks whether the given matrix contains NaN entries.
* @param m Input matrix.
* @return True if m contrains NaN entries.
*/
inline bool containsNaNs(const cv::Mat& m)
{
    return cv::sum(cv::Mat(m != m))[0] > 0;
}

/**
 * @brief Wrapper around the OpenCV PnP function that returns a zero pose in case PnP fails. See also documentation of cv::solvePnP.
 * @param objPts List of 3D points.
 * @param imgPts Corresponding 2D points.
 * @param camMat Calibration matrix of the camera.
 * @param distCoeffs Distortion coefficients.
 * @param rot Output parameter. Camera rotation.
 * @param trans Output parameter. Camera translation.
 * @param extrinsicGuess If true uses input rot and trans as initialization.
 * @param methodFlag Specifies the PnP algorithm to be used.
 * @return True if PnP succeeds.
 */
inline bool safeSolvePnP(
    const std::vector<cv::Point3f>& objPts,
    const std::vector<cv::Point2f>& imgPts,
    const cv::Mat& camMat,
    const cv::Mat& distCoeffs,
    cv::Mat& rot,
    cv::Mat& trans,
    bool extrinsicGuess,
    int methodFlag)
{
    if(rot.type() == 0) rot = cv::Mat_::zeros(1, 3);
    if(trans.type() == 0) trans= cv::Mat_::zeros(1, 3);

    if(!cv::solvePnP(objPts, imgPts, camMat, distCoeffs, rot, trans, extrinsicGuess,methodFlag))
    {
        rot = cv::Mat_<double>::zeros(1, 3);
        trans = cv::Mat_<double>::zeros(1, 3);
        return false;
    }
    return true;
}

/**
 * @brief Calculate the Shannon entropy of a discrete distribution.
 * @param dist Discrete distribution. Probability per entry, should sum to 1.
 * @return  Shannon entropy.
 */
double entropy(const std::vector<double>& dist)
{
    double e = 0;
    for(unsigned i = 0; i < dist.size(); i++)
	if(dist[i] > 0)
	    e -= dist[i] * std::log2(dist[i]);
    
    return e;
}

/**
 * @brief Draws an entry of a discrete distribution according to the given probabilities.
 *
 * If randomDraw is false in the properties, this function will return the entry with the max. probability.
 *
 * @param probs Discrete distribution. Probability per entry, should sum to 1.
 * @return Chosen entry.
 */
int draw(const std::vector<double>& probs)
{
    std::map<double, int> cumProb;
    double probSum = 0;
    double maxProb = -1;
    double maxIdx = 0; 
    
    for(unsigned idx = 0; idx < probs.size(); idx++)
    {
	if(probs[idx] < EPS) continue;
	
	probSum += probs[idx];
	cumProb[probSum] = idx;
	
	if(maxProb < 0 || probs[idx] > maxProb)
	{
	    maxProb = probs[idx];
	    maxIdx = idx;
	}
    }
    
    if(GlobalProperties::getInstance()->tP.randomDraw)
      return cumProb.upper_bound(drand(0, probSum))->second;
    else
      return maxIdx;
}

/**
 * @brief Calculates the expected loss of a list of poses with associated probabilities.
 * @param gt Ground truth pose.
 * @param hyps List of estimated poses.
 * @param probs List of probabilities associated with the estimated poses.
 * @param losses Output parameter. List of losses for each estimated pose.
 * @return Expectation of loss.
 */
double expectedMaxLoss(
    const jp::cv_trans_t& gt,
    const std::vector<jp::cv_trans_t>& hyps,
    const std::vector<double>& probs,
    std::vector<double>& losses)
{
    double loss = 0;
    losses.resize(hyps.size());
    
    for(unsigned i = 0; i < hyps.size(); i++)
    {
        losses[i] = maxLoss(gt, hyps.at(i));
        loss += probs[i] * losses[i];
    }
    
    return loss;
}

/**
 * @brief Calculates the Jacobean of the PNP function w.r.t. the object coordinate inputs.
 *
 * PNP is treated as a n x 3 -> 6 fnuction, i.e. it takes n 3D coordinates and maps them to a 6D pose.
 * The Jacobean is therefore 6x3n. The Jacobean is calculated using central differences.
 *
 * @param imgPts List of 2D points.
 * @param objPts List of corresponding 3D points.
 * @param eps Epsilon used in central differences approximation.
 * @return 6x3n Jacobean matrix of partial derivatives.
 */
cv::Mat_<double> dPNP(    
    const std::vector<cv::Point2f>& imgPts,
    std::vector<cv::Point3f> objPts,
    float eps = 0.001f)
{
    int pnpMethod = (imgPts.size() == 4) ? CV_P3P : CV_ITERATIVE;

    //in case of P3P the 4th point is needed to resolve ambiguities, its derivative is zero
    int effectiveObjPoints = (pnpMethod == CV_P3P) ? 3 : objPts.size();

    cv::Mat_<float> camMat = GlobalProperties::getInstance()->getCamMat();
    cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(6, objPts.size() * 3);
    bool success;
    
    // central differences
    for(unsigned i = 0; i < effectiveObjPoints; i++)
    for(unsigned j = 0; j < 3; j++)
    {
        if(j == 0) objPts[i].x += eps;
        else if(j == 1) objPts[i].y += eps;
        else if(j == 2) objPts[i].z += eps;

        // forward step
        jp::cv_trans_t fStep;
        success = safeSolvePnP(objPts, imgPts, camMat, cv::Mat(), fStep.first, fStep.second, false, pnpMethod);

        if(!success)
            return cv::Mat_<double>::zeros(6, objPts.size() * 3);

        if(j == 0) objPts[i].x -= 2 * eps;
        else if(j == 1) objPts[i].y -= 2 * eps;
        else if(j == 2) objPts[i].z -= 2 * eps;

        // backward step
        jp::cv_trans_t bStep;
        success = safeSolvePnP(objPts, imgPts, camMat, cv::Mat(), bStep.first, bStep.second, false, pnpMethod);

        if(!success)
            return cv::Mat_<double>::zeros(6, objPts.size() * 3);

        if(j == 0) objPts[i].x += eps;
        else if(j == 1) objPts[i].y += eps;
        else if(j == 2) objPts[i].z += eps;

        // gradient calculation
        fStep.first = (fStep.first - bStep.first) / (2 * eps);
        fStep.second = (fStep.second - bStep.second) / (2 * eps);

        fStep.first.copyTo(jacobean.col(i * 3 + j).rowRange(0, 3));
        fStep.second.copyTo(jacobean.col(i * 3 + j).rowRange(3, 6));

        if(containsNaNs(jacobean.col(i * 3 + j)))
            return cv::Mat_<double>::zeros(6, objPts.size() * 3);
    }

    return jacobean;
}

/**
 * @brief Calculate the average of all matrix entries.
 * @param mat Input matrix.
 * @return Average of entries.
 */
double getAvg(const cv::Mat_<double>& mat)
{
    double avg = 0;
    
    for(unsigned x = 0; x < mat.cols; x++)
    for(unsigned y = 0; y < mat.rows; y++)
    {
	avg += std::abs(mat(y, x));
    }
    
    return avg / mat.cols / mat.rows;
}

/**
 * @brief Return the maximum entry of the given matrix.
 * @param mat Input matrix.
 * @return Maximum entry.
 */
double getMax(const cv::Mat_<double>& mat)
{
    double m = -1;
    
    for(unsigned x = 0; x < mat.cols; x++)
    for(unsigned y = 0; y < mat.rows; y++)
    {
	double val = std::abs(mat(y, x));
	if(m < 0 || val > m)
	  m = val;
    }
    
    return m;
}

/**
 * @brief Return the median of all entries of the given matrix.
 * @param mat Input matrix.
 * @return Median entry.
 */
double getMed(const cv::Mat_<double>& mat)
{
    std::vector<double> vals;
    
    for(unsigned x = 0; x < mat.cols; x++)
    for(unsigned y = 0; y < mat.rows; y++)
	vals.push_back(std::abs(mat(y, x)));

    std::sort(vals.begin(), vals.end());
    
    return vals[vals.size() / 2];
}

/**
 * @brief Transform an RGB image to a floating point CNN input map.
 *
 * The image will be cropped to CNN input size.
 * In training mode, the input will be randomly shifted by small amounts, depending on the subsampling in the CNN output.
 * The method also creates a sampling map which maps each output of the CNN to a 2D input position in the original RGB image.
 *
 * @param img Input RGB image.
 * @param sampling Map that contains for each position in the CNN output the corresponding position in the RGB input (use to establish 2D-3D correspondences).
 * @param training True for training mode.
 * @return
 */
cv::Mat_<cv::Vec3f> getImgMap(const jp::img_bgr_t& img, cv::Mat_<cv::Point2i>& sampling, bool training)
{
    GlobalProperties* gp = GlobalProperties::getInstance();

    int cnnInputW = gp->getCNNInputDimX();
    int cnnInputH = gp->getCNNInputDimY();
    int cnnOutputW = gp->getCNNOutputDimX();
    int cnnOutputH = gp->getCNNOutputDimY();
    int cnnSubSampling = gp->dP.cnnSubSample;

    cv::Mat_<cv::Vec3f> imgMap(cnnInputH, cnnInputW);
    sampling = cv::Mat_<cv::Point2i>(cnnOutputH, cnnOutputW);

    int offsetX = img.cols - cnnInputW;
    int offsetY = img.rows - cnnInputH;

    if(training)
    {
        // random shift
        offsetX = irand(0, offsetX);
        offsetY = irand(0, offsetY);
    }
    else
    {
        // crop at the center
        offsetX /= 2;
        offsetY /= 2;
    }

    // crop image
    for(unsigned x = 0; x < cnnInputW; x++)
    for(unsigned y = 0; y < cnnInputH; y++)
    {
        imgMap(y, x) = img(y + offsetY, x + offsetX);
    }

    // create sampling map
    for(unsigned x = 0; x < sampling.cols; x++)
    for(unsigned y = 0; y < sampling.rows; y++)
    {
        sampling(y, x) = cv::Point2i(
            offsetX + x * cnnSubSampling + cnnSubSampling / 2,
            offsetY + y * cnnSubSampling + cnnSubSampling / 2);
    }

    return imgMap;
}

/**
 * @brief Process a RGB image with the object coordinate CNN.
 * @param colorData Input RGB image.
 * @param sampling Output paramter. Subsampling information. Each 2D location contains the pixel location in the original RGB image (needed again for backward pass).
 * @param imgMaps Output parameter. RGB image transformed to CNN input maps (needed again for backward pass).
 * @param training True if training mode (controls cropping if input image).
 * @param state Lua state for access to the object coordinate CNN.
 * @return Object coordinate estimation (sub sampled).
 */
jp::img_coord_t getCoordImg(
    const jp::img_bgr_t& colorData, 
    cv::Mat_<cv::Point2i>& sampling,
    std::vector<cv::Mat_<cv::Vec3f>>& imgMaps,
    bool training,
    lua_State* state)
{
    StopWatch stopW;

    imgMaps.resize(1);
    imgMaps[0] = getImgMap(colorData, sampling, training);

    // forward pass
    std::vector<cv::Vec3f> prediction = forward(imgMaps, sampling, state);

    // reorganize
    jp::img_coord_t modeImg =
        jp::img_coord_t::zeros(sampling.rows, sampling.cols);

    for(unsigned i = 0; i < prediction.size(); i++)
    {
   	int x = i % modeImg.cols;
	int y = i / modeImg.cols;   
      
        modeImg(y, x) = prediction[i];
    }
    
    std::cout << "CNN prediction took " << stopW.stop() / 1000 << "s." << std::endl;
    
    return modeImg;
}

/**
 * @brief Calculate an image of reprojection errors for the given object coordinate prediction and the given pose.
 * @param hyp Pose estimate.
 * @param objectCoordinates Object coordinate estimate.
 * @param sampling Subsampling of the input image.
 * @param camMat Calibration matrix of the camera.
 * @return Image of reprojectiob errors.
 */
cv::Mat_<float> getDiffMap(
  const jp::cv_trans_t& hyp,
  const jp::img_coord_t& objectCoordinates,
  const cv::Mat_<cv::Point2i>& sampling,
  const cv::Mat& camMat)
{
    cv::Mat_<float> diffMap(sampling.size());
  
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> projections;	
    std::vector<cv::Point2f> points2D;
    std::vector<cv::Point2f> sources2D;
    
    // collect 2D-3D correspondences
    for(unsigned x = 0; x < sampling.cols; x++)
    for(unsigned y = 0; y < sampling.rows; y++)
    {
        // get 2D location of the original RGB frame
	cv::Point2f pt2D(sampling(y, x).x, sampling(y, x).y);
	
        // get associated 3D object coordinate prediction
	points3D.push_back(cv::Point3f(
	    objectCoordinates(y, x)(0), 
	    objectCoordinates(y, x)(1), 
	    objectCoordinates(y, x)(2)));
	points2D.push_back(pt2D);
	sources2D.push_back(cv::Point2f(x, y));
    }
    
    if(points3D.empty()) return diffMap;

    // project object coordinate into the image using the given pose
    cv::projectPoints(points3D, hyp.first, hyp.second, camMat, cv::Mat(), projections);
    
    // measure reprojection errors
    for(unsigned p = 0; p < projections.size(); p++)
    {
	cv::Point2f curPt = points2D[p] - projections[p];
	float l = std::min(cv::norm(curPt), CNN_OBJ_MAXINPUT);
	diffMap(sources2D[p].y, sources2D[p].x) = l;
    }

    return diffMap;    
}

/**
 * @brief Project a 3D point into the image an measures the reprojection error.
 * @param pt Ground truth 2D location.
 * @param obj 3D point.
 * @param hyp Pose estimate.
 * @param camMat Calibration matrix of the camera.
 * @return Reprojection error in pixels.
 */
float project(const cv::Point2f& pt, const cv::Point3f& obj, const jp::cv_trans_t hyp, const cv::Mat& camMat)
{
    double f = camMat.at<float>(0, 0);
    double ppx = camMat.at<float>(0, 2);
    double ppy = camMat.at<float>(1, 2);
    
    //transform point
    cv::Mat objMat = cv::Mat(obj);
    objMat.convertTo(objMat, CV_64F);
    
    cv::Mat rot;
    cv::Rodrigues(hyp.first, rot);

    objMat = rot * objMat + hyp.second;
    
    // project
    double px = f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) + ppx;
    double py = f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) + ppy;
    
    //std::cout << "Projected position: " << px << ", " << py << std::endl;
    
    // return error
    return std::min(std::sqrt((pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py)), CNN_OBJ_MAXINPUT);
}

/**
 * @brief Calculates the Jacobean of the projection function w.r.t the given 3D point, ie. the function has the form 3 -> 1
 * @param pt Ground truth 2D location.
 * @param obj 3D point.
 * @param hyp Pose estimate.
 * @param camMat Calibration matrix of the camera.
 * @return 1x3 Jacobean matrix of partial derivatives.
 */
cv::Mat_<double> dProjectdObj(const cv::Point2f& pt, const cv::Point3f& obj, const jp::cv_trans_t hyp, const cv::Mat& camMat)
{
    double f = camMat.at<float>(0, 0);
    double ppx = camMat.at<float>(0, 2);
    double ppy = camMat.at<float>(1, 2);
    
    //transform point
    cv::Mat objMat = cv::Mat(obj);
    objMat.convertTo(objMat, CV_64F);

    cv::Mat rot;
    cv::Rodrigues(hyp.first, rot);
    
    objMat = rot * objMat + hyp.second;
    
    if(std::abs(objMat.at<double>(2, 0)) < EPS) // prevent division by zero
        return cv::Mat_<double>::zeros(1, 3);

    // project
    double px = f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) + ppx;
    double py = f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) + ppy;
    
    // calculate error
    double err = std::sqrt((pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py));
    
    // early out if projection error is above threshold
    if(err > CNN_OBJ_MAXINPUT)
	return cv::Mat_<double>::zeros(1, 3);
    
    err += EPS; // avoid dividing by zero
    
    // derivative in x direction of obj coordinate
    double pxdx = f * rot.at<double>(0, 0) / objMat.at<double>(2, 0) - f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 0);
    double pydx = f * rot.at<double>(1, 0) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 0);
    double dx = 0.5 / err * (2 * (pt.x - px) * -pxdx + 2 * (pt.y - py) * -pydx);

    // derivative in x direction of obj coordinate
    double pxdy = f * rot.at<double>(0, 1) / objMat.at<double>(2, 0) - f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 1);
    double pydy = f * rot.at<double>(1, 1) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 1);
    double dy = 0.5 / err * (2 * (pt.x - px) * -pxdy + 2 * (pt.y - py) * -pydy);
    
    // derivative in x direction of obj coordinate
    double pxdz = f * rot.at<double>(0, 2) / objMat.at<double>(2, 0) - f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 2);
    double pydz = f * rot.at<double>(1, 2) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 2);
    double dz = 0.5 / err * (2 * (pt.x - px) * -pxdz + 2 * (pt.y - py) * -pydz);	
    
    cv::Mat_<double> jacobean(1, 3);
    jacobean(0, 0) = dx;
    jacobean(0, 1) = dy;
    jacobean(0, 2) = dz;
    
    return jacobean;
}

/**
 * @brief Calculates the Jacobean of the projection function w.r.t the given 6D pose, ie. the function has the form 6 -> 1
 * @param pt Ground truth 2D location.
 * @param obj 3D point.
 * @param hyp Pose estimate.
 * @param camMat Calibration matrix of the camera.
 * @return 1x6 Jacobean matrix of partial derivatives.
 */
cv::Mat_<double> dProjectdHyp(const cv::Point2f& pt, const cv::Point3f& obj, const jp::cv_trans_t hyp, const cv::Mat& camMat)
{
    double f = camMat.at<float>(0, 0);
    double ppx = camMat.at<float>(0, 2);
    double ppy = camMat.at<float>(1, 2);
    
    //transform point
    cv::Mat objMat = cv::Mat(obj);
    objMat.convertTo(objMat, CV_64F);
    
    cv::Mat rot, dRdH;
    cv::Rodrigues(hyp.first, rot, dRdH);
    dRdH = dRdH.t();

    cv::Mat eyeMat = rot * objMat + hyp.second;
    
    if(std::abs(eyeMat.at<double>(2, 0)) < EPS) // prevent division by zero
        return cv::Mat_<double>::zeros(1, 6);

    // project
    double px = f * eyeMat.at<double>(0, 0) / eyeMat.at<double>(2, 0) + ppx; // flip x because of reasons (to conform with OpenCV implementation)
    double py = f * eyeMat.at<double>(1, 0) / eyeMat.at<double>(2, 0) + ppy;
    
    // calculate error
    double err = std::sqrt((pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py));

    // early out if projection error is above threshold
    if(err > CNN_OBJ_MAXINPUT)
	return cv::Mat_<double>::zeros(1, 6);
    
    err += EPS; // avoid dividing by zero
    
    // derivative of the error wrt to projection
    cv::Mat_<double> dNdP = cv::Mat_<double>::zeros(1, 2);
    dNdP(0, 0) = -1 / err * (pt.x - px);
    dNdP(0, 1) = -1 / err * (pt.y - py);
    
    // derivative of projection function wrt rotation matrix
    cv::Mat_<double> dPdR = cv::Mat_<double>::zeros(2, 9);
    dPdR.row(0).colRange(0, 3) = f * objMat.t() / eyeMat.at<double>(2, 0);
    dPdR.row(1).colRange(3, 6) = f * objMat.t() / eyeMat.at<double>(2, 0);
    dPdR.row(0).colRange(6, 9) = -f * eyeMat.at<double>(0, 0) / eyeMat.at<double>(2, 0) / eyeMat.at<double>(2, 0) * objMat.t();
    dPdR.row(1).colRange(6, 9) = -f * eyeMat.at<double>(1, 0) / eyeMat.at<double>(2, 0) / eyeMat.at<double>(2, 0) * objMat.t();
        
    // combined derivative of the error wrt the rodriguez vector
    cv::Mat_<double> dNdH = dNdP * dPdR * dRdH;
    
    // derivative of projection wrt the translation vector
    cv::Mat_<double> dPdT = cv::Mat_<double>::zeros(2, 3);
    dPdT(0, 0) = f / eyeMat.at<double>(2, 0);
    dPdT(1, 1) = f / eyeMat.at<double>(2, 0);
    dPdT(0, 2) = -f * eyeMat.at<double>(0, 0) / eyeMat.at<double>(2, 0) / eyeMat.at<double>(2, 0);
    dPdT(1, 2) = -f * eyeMat.at<double>(1, 0) / eyeMat.at<double>(2, 0) / eyeMat.at<double>(2, 0);
    
    // combined derivative of error wrt the translation vector 
    cv::Mat_<double> dNdT = dNdP * dPdT;
    
    cv::Mat_<double> jacobean(1, 6);
    dNdH.copyTo(jacobean.colRange(0, 3));
    dNdT.copyTo(jacobean.colRange(3, 6));
    return jacobean;
}

/**
 * @brief Applies soft max to the given list of scores.
 * @param scores List of scores.
 * @return Soft max distribution (sums to 1)
 */
std::vector<double> softMax(const std::vector<double>& scores)
{
    double maxScore = 0;
    for(unsigned i = 0; i < scores.size(); i++)
        if(i == 0 || scores[i] > maxScore) maxScore = scores[i];
	
    std::vector<double> sf(scores.size());
    double sum = 0.0;
    
    for(unsigned i = 0; i < scores.size(); i++)
    {
	sf[i] = std::exp(scores[i] - maxScore);
	sum += sf[i];
    }
    for(unsigned i = 0; i < scores.size(); i++)
    {
	sf[i] /= sum;
// 	std::cout << "score: " << scores[i] << ", prob: " << sf[i] << std::endl;
    }
    
    return sf;
}

/**
 * @brief Calculates the Jacobean matrix of the function that maps n estimated object coordinates to a score, ie. the function has the form n x 3 -> 1. Returns one Jacobean matrix per hypothesis.
 * @param estObj Object coordinate estimation.
 * @param sampling Sub sampling of the RGB image.
 * @param points List of minimal sets. Each one (4 correspondences) defines one hypothesis.
 * @param stateObj Lua state for access to the score CNN.
 * @param jacobeans Output paramter. List of Jacobean matrices. One 1 x 3n matrix per pose hypothesis.
 * @param scoreOutputGradients Gradients w.r.t the score i.e. the gradients of the output of the score CNN.
 */
void dScore(
    jp::img_coord_t estObj,
    const cv::Mat_<cv::Point2i>& sampling,
    const std::vector<std::vector<cv::Point2i>>& points,
    lua_State* stateObj,
    std::vector<cv::Mat_<double>>& jacobeans,
    const std::vector<double>& scoreOutputGradients)
{
    GlobalProperties* gp = GlobalProperties::getInstance();
    cv::Mat_<float> camMat = gp->getCamMat();
  
    int hypCount = points.size();
    
    std::vector<std::vector<cv::Point2f>> imgPts(hypCount);
    std::vector<std::vector<cv::Point3f>> objPts(hypCount);
    std::vector<jp::cv_trans_t> hyps(hypCount);
    std::vector<cv::Mat_<float>> diffMaps(hypCount);
    
    #pragma omp parallel for
    for(unsigned h = 0; h < hypCount; h++)
    {
        for(unsigned i = 0; i < points[h].size(); i++)
        {
            int x = points[h][i].x;
            int y = points[h][i].y;
	  
            imgPts[h].push_back(sampling(y, x));
            objPts[h].push_back(cv::Point3f(estObj(y, x)));
        }
      
        // calculate hypothesis
        jp::cv_trans_t cvHyp;
        safeSolvePnP(objPts[h], imgPts[h], camMat, cv::Mat(), cvHyp.first, cvHyp.second, false, CV_P3P);
        hyps[h] = cvHyp;
	    
        // calculate projection errors
        diffMaps[h] = getDiffMap(cvHyp, estObj, sampling, camMat);
    }
    
    std::vector<cv::Mat_<double>> dDiffMaps;
    backward(diffMaps, stateObj, scoreOutputGradients, dDiffMaps);

    jacobeans.resize(hypCount);

    #pragma omp parallel for
    for(unsigned h = 0; h < hypCount; h++)
    {        
        cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, estObj.cols * estObj.rows * 3);
        jacobeans[h] = jacobean;

        if(cv::norm(dDiffMaps[h]) < EPS) continue;

        // accumulate derivate of score wrt the object coordinates that are used to calculate the pose
        cv::Mat_<double> supportPointGradients = cv::Mat_<double>::zeros(1, 12);

        cv::Mat_<double> dHdO = dPNP(imgPts[h], objPts[h]); // 6x12

        for(unsigned x = 0; x < dDiffMaps[h].cols; x++)
        for(unsigned y = 0; y < dDiffMaps[h].rows; y++)
        {
            cv::Point2f pt(sampling(y, x).x, sampling(y, x).y);
            cv::Point3f obj(estObj(y, x));
	  
            // account for the direct influence of all object coordinates in the score
            cv::Mat_<double> dPdO = dProjectdObj(pt, obj, hyps[h], camMat);
            dPdO *= dDiffMaps[h](y, x);
            dPdO.copyTo(jacobean.colRange(x * dDiffMaps[h].rows * 3 + y * 3, x * dDiffMaps[h].rows * 3 + y * 3 + 3));
	    
            // account for the indirect influence of the object coorindates that are used to calculate the pose
            cv::Mat_<double> dPdH = dProjectdHyp(sampling(y, x), cv::Point3f(estObj(y, x)), hyps[h], camMat);
            supportPointGradients += dDiffMaps[h](y, x) * dPdH * dHdO;
        }

        // add the accumulated derivatives for the object coordinates that are used to calculate the pose
        for(unsigned i = 0; i < points[h].size(); i++)
        {
            unsigned x = points[h][i].x;
            unsigned y = points[h][i].y;
	    
            jacobean.colRange(x * dDiffMaps[h].rows * 3 + y * 3, x * dDiffMaps[h].rows * 3 + y * 3 + 3) += supportPointGradients.colRange(i * 3, i * 3 + 3);
        }
    }
}

/**
 * @brief Calculates the Jacobean matrix of the function that maps n estimated object coordinates to a soft max score, ie. the function has the form n x 3 -> 1. Returns one Jacobean matrix per hypothesis.
 *
 * This is the Soft maxed version of dScore (see above).
 *
 * @param estObj Object coordinate estimation.
 * @param sampling Sub sampling of the RGB image.
 * @param points List of minimal sets. Each one (4 correspondences) defines one hypothesis.
 * @param losses Loss measured for the hypotheses given by the points parameter.
 * @param sfScores Soft max probabilities for the hypotheses given by the points parameter.
 * @param stateObj Lua state for access to the score CNN.
 * @return List of Jacobean matrices. One 1 x 3n matrix per pose hypothesis.
 */
std::vector<cv::Mat_<double>> dSMScore(
    jp::img_coord_t estObj,
    const cv::Mat_<cv::Point2i>& sampling,
    const std::vector<std::vector<cv::Point2i>>& points,
    const std::vector<double>& losses,
    const std::vector<double>& sfScores,
    lua_State* stateObj)
{
    // assemble the gradients wrt the scores, ie the gradients of soft max function
    std::vector<double> scoreOutputGradients(points.size());
        
    for(unsigned i = 0; i < points.size(); i++)
    {
        scoreOutputGradients[i] = sfScores[i] * losses[i];
        for(unsigned j = 0; j < points.size(); j++)
            scoreOutputGradients[i] -= sfScores[i] * sfScores[j] * losses[j];
    }
 
    // calculate gradients of the score function
    std::vector<cv::Mat_<double>> jacobeans;
    dScore(estObj, sampling, points, stateObj, jacobeans, scoreOutputGradients);

    // data conversion
    for(unsigned i = 0; i < jacobeans.size(); i++)
    {
        // reorder to points row first into rows
        cv::Mat_<double> reformat(estObj.cols * estObj.rows, 3);
	
        for(unsigned x = 0; x < estObj.cols; x++)
        for(unsigned y = 0; y < estObj.rows; y++)
        {
            cv::Mat_<double> patchGrad = jacobeans[i].colRange(
              x * estObj.rows * 3 + y * 3,
              x * estObj.rows * 3 + y * 3 + 3);
	    
            patchGrad.copyTo(reformat.row(y * estObj.cols + x));
        }
	
        jacobeans[i] = reformat;
    }
    
    return jacobeans;
}

/**
 * @brief Processes a frame, ie. takes object coordinates, estimates poses, selects the best one and measures the error.
 *
 * This function performs the forward pass of DSAC but also calculates many intermediate results
 * for the backward pass (ie it can be made faster if one cares only about the forward pass).
 *
 * @param poseGT Ground truth pose (for evaluation only).
 * @param stateObj Lua state for access to the score CNN.
 * @param objHyps Number of hypotheses to be drawn.
 * @param camMat Calibration parameters of the camera.
 * @param inlierThreshold2D Inlier threshold in pixels.
 * @param refSteps Max. refinement steps (iterations).
 * @param expectedLoss Output paramter. Expectation of loss of the discrete hypothesis distributions.
 * @param sfEntropy Output parameter. Shannon entropy of the soft max distribution of hypotheses.
 * @param correct Output parameter. Was the final, selected hypothesis correct?
 * @param refHyps Output parameter. List of refined hypotheses sampled for the given image.
 * @param sfScores Output parameter. Soft max distribution for the sampled hypotheses.
 * @param estObj Output parameter. Estimated object coordinates (subsampling of the complete image).
 * @param sampling Output parameter. Subsampling of the RGB image.
 * @param sampledPoints Output parameter. List of initial 2D pixel locations of the subsampled input RGB image. 4 pixels per hypothesis.
 * @param losses Output parameter. List of losses of the sampled hypotheses.
 * @param inlierMaps Output parameter. Maps indicating which pixels of the subsampled input image have been inliers in the last step of hypothesis refinement, one map per hypothesis.
 * @param tErr Output parameter. Translational (in m) error of the final, selected hypothesis.
 * @param rotErr Output parameter. Rotational error of the final, selected hypothesis.
 * @param hypIdx Output parameter. Index of the final, selected hypothesis.
 * @param training True if training mode. Controls whether all hypotheses are refined or just the selected one.
 */
void processImage(
    const jp::cv_trans_t& hypGT,
    lua_State* stateObj,
    int objHyps,
    const cv::Mat& camMat,
    int inlierThreshold2D,
    int refSteps,
    double& expectedLoss,
    double& sfEntropy,
    bool& correct,
    std::vector<jp::cv_trans_t>& refHyps,
    std::vector<double>& sfScores,
    const jp::img_coord_t& estObj,
    const cv::Mat_<cv::Point2i>& sampling,
    std::vector<std::vector<cv::Point2i>>& sampledPoints,
    std::vector<double>& losses,
    std::vector<cv::Mat_<int>>& inlierMaps,
    double& tErr,
    double& rotErr,
    int& hypIdx,
    bool training = true)
{
    std::cout << BLUETEXT("Sampling " << objHyps << " hypotheses.") << std::endl;
    StopWatch stopW;

    sampledPoints.resize(objHyps);    // keep track of the points each hypothesis is sampled from
    refHyps.resize(objHyps);
    std::vector<std::vector<cv::Point2f>> imgPts(objHyps);
    std::vector<std::vector<cv::Point3f>> objPts(objHyps);

    // sample hypotheses
    #pragma omp parallel for
    for(unsigned h = 0; h < refHyps.size(); h++)
    while(true)
    {
	std::vector<cv::Point2f> projections;
	cv::Mat_<uchar> alreadyChosen = cv::Mat_<uchar>::zeros(estObj.size());
	imgPts[h].clear();
        objPts[h].clear();
	sampledPoints[h].clear();

        for(int j = 0; j < 4; j++)
	{
            // 2D location in the subsampled image
	    int x = irand(0, estObj.cols);
	    int y = irand(0, estObj.rows);
	    
	    if(alreadyChosen(y, x) > 0)
	    {
                j--;
                continue;
	    }
	    
	    alreadyChosen(y, x) = 1;
	    
            imgPts[h].push_back(sampling(y, x)); // 2D location in the original RGB image
            objPts[h].push_back(cv::Point3f(estObj(y, x))); // 3D object coordinate
            sampledPoints[h].push_back(cv::Point2i(x, y)); // 2D pixel location in the subsampled image
	}

        if(!safeSolvePnP(objPts[h], imgPts[h], camMat, cv::Mat(), refHyps[h].first, refHyps[h].second, false, CV_P3P))
        {
            continue;
        }

        cv::projectPoints(objPts[h], refHyps[h].first, refHyps[h].second, camMat, cv::Mat(), projections);

        // check reconstruction, 4 sampled points should be reconstructed perfectly
        bool foundOutlier = false;
        for(unsigned j = 0; j < imgPts[h].size(); j++)
        {
            if(cv::norm(imgPts[h][j] - projections[j]) < inlierThreshold2D)
                continue;
            foundOutlier = true;
            break;
        }
        if(foundOutlier)
            continue;
        else
            break;
    }	

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
    std::cout << BLUETEXT("Calculating scores.") << std::endl;

    // compute reprojection error images
    std::vector<cv::Mat_<float>> diffMaps(objHyps);
    #pragma omp parallel for 
    for(unsigned h = 0; h < refHyps.size(); h++)
        diffMaps[h] = getDiffMap(refHyps[h], estObj, sampling, camMat);

    // execute score script to get hypothesis scores
    std::vector<double> scores = forward(diffMaps, stateObj);
    
    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
    std::cout << BLUETEXT("Drawing final Hypothesis.") << std::endl;	
    
    // apply soft max to scores to get a distribution
    sfScores = softMax(scores);
    sfEntropy = entropy(sfScores); // measure distribution entropy
    hypIdx = draw(sfScores); // select winning hypothesis

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
    std::cout << BLUETEXT("Refining poses:") << std::endl;
    
    // collect inliers
    inlierMaps.resize(refHyps.size());
    
    double convergenceThresh = 0.01; // stop refinement if 6D pose vector converges

    #pragma omp parallel for
    for(unsigned h = 0; h < refHyps.size(); h++)
    {
        if(!training && hypIdx != h)
            continue; // in test mode only refine selected hypothesis

        cv::Mat_<float> localDiffMap = diffMaps[h];

        // refine current hypothesis
	for(unsigned rStep = 0; rStep < refSteps; rStep++)
        {
            // collect inliers
	    std::vector<cv::Point2f> localImgPts;
	    std::vector<cv::Point3f> localObjPts; 
            cv::Mat_<int> localInlierMap = cv::Mat_<int>::zeros(diffMaps[h].size());
	    
            for(unsigned x = 0; x < localDiffMap.cols; x++)
            for(unsigned y = 0; y < localDiffMap.rows; y++)
	    {
    		if(localDiffMap(y, x) < inlierThreshold2D)
		{
		    localImgPts.push_back(sampling(y, x));
		    localObjPts.push_back(cv::Point3f(estObj(y, x)));
                    localInlierMap(y, x) = 1;
		}
            }

            if(localImgPts.size() < 4)
                break;

            // recalculate pose
	    jp::cv_trans_t hypUpdate;
	    hypUpdate.first = refHyps[h].first.clone();
	    hypUpdate.second = refHyps[h].second.clone();

            if(!safeSolvePnP(localObjPts, localImgPts, camMat, cv::Mat(), hypUpdate.first, hypUpdate.second, true, (localImgPts.size() > 4) ? CV_ITERATIVE : CV_P3P))
                break; //abort if PnP fails

            if(maxLoss(hypUpdate, refHyps[h]) < convergenceThresh)
                break; // convergned

	    refHyps[h] = hypUpdate;
            inlierMaps[h] = localInlierMap;

	    // recalculate pose errors
	    localDiffMap = getDiffMap(refHyps[h], estObj, sampling, camMat);
	}
    }

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
    std::cout << BLUETEXT("Final Result:") << std::endl;
    
    // evaluated poses
    expectedLoss = expectedMaxLoss(hypGT, refHyps, sfScores, losses);
    std::cout << "Loss of winning hyp: " << maxLoss(hypGT, refHyps[hypIdx]) << ", prob: " << sfScores[hypIdx] << ", expected loss: " << expectedLoss << std::endl;

    // we measure error of inverted poses (because we estimate scene poses, not camera poses)
    jp::cv_trans_t invHypGT = getInvHyp(hypGT);
    jp::cv_trans_t invHypEst = getInvHyp(refHyps[hypIdx]);

    rotErr = calcAngularDistance(invHypGT, invHypEst);
    tErr = cv::norm(invHypEst.second - invHypGT.second);

    correct = false;
    if(rotErr < 5 && tErr < 0.05)
    {
        std::cout << GREENTEXT("Rotation Err: " << rotErr << "deg, Translation Err: " << tErr * 100 << "cm") << std::endl << std::endl;
        correct = true;
    }
    else
        std::cout << REDTEXT("Rotation Err: " << rotErr << "deg, Translation Err: " << tErr * 100 << "cm") << std::endl << std::endl;
}

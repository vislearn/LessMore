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

#define MAXLOSS 10000000.0

/**
 * @brief Inverts a given transformation.
 * @param hyp Input transformation.
 * @return Inverted transformation.
 */
jp::cv_trans_t getInvHyp(const jp::cv_trans_t& hyp)
{
    cv::Mat_<double> hypR, trans = cv::Mat_<float>::eye(4, 4);
    cv::Rodrigues(hyp.first, hypR);

    hypR.copyTo(trans.rowRange(0,3).colRange(0,3));
    trans(0, 3) = hyp.second.at<double>(0, 0);
    trans(1, 3) = hyp.second.at<double>(0, 1);
    trans(2, 3) = hyp.second.at<double>(0, 2);

    trans = trans.inv();

    jp::cv_trans_t invHyp;
    cv::Rodrigues(trans.rowRange(0,3).colRange(0,3), invHyp.first);
    invHyp.second = cv::Mat_<double>(1, 3);
    invHyp.second.at<double>(0, 0) = trans(0, 3);
    invHyp.second.at<double>(0, 1) = trans(1, 3);
    invHyp.second.at<double>(0, 2) = trans(2, 3);

    return invHyp;
}

/**
 * @brief Calculates the rotational distance in degree between two transformations.
 * Translation will be ignored.
 *
 * @param h1 Transformation 1.
 * @param h2 Transformation 2.
 * @return Angle in degree.
 */
double calcAngularDistance(const jp::cv_trans_t& h1, const jp::cv_trans_t& h2)
{
    cv::Mat r1, r2;
    cv::Rodrigues(h1.first, r1);
    cv::Rodrigues(h2.first, r2);

    cv::Mat rotDiff= r2 * r1.t();
    double trace = cv::trace(rotDiff)[0];

    trace = std::min(3.0, std::max(-1.0, trace));
    return 180*acos((trace-1.0)/2.0)/CV_PI;
}

/**
 * @brief Maximum of translational error (cm) and rotational error (deg) between two pose hypothesis.
 * @param h1 Pose 1.
 * @param h2 Pose 2.
 * @return Loss.
 */
double maxLoss(const jp::cv_trans_t& h1, const jp::cv_trans_t& h2)
{
    // measure loss of inverted poses (camera pose instead of scene pose)
    jp::cv_trans_t invH1 = getInvHyp(h1);
    jp::cv_trans_t invH2 = getInvHyp(h2);

    double rotErr = calcAngularDistance(invH1, invH2);
    double tErr = cv::norm(invH1.second - invH2.second);

    return std::min(std::max(rotErr, tErr * 100), MAXLOSS);
}

/**
 * @brief Calculate the derivative of the loss w.r.t. the estimated pose.
 * @param est Estimated pose (6 DoF).
 * @param gt Ground truth pose (6 DoF).
 * @return 1x6 Jacobean.
 */
cv::Mat_<double> dLossMax(const jp::cv_trans_t& est, const jp::cv_trans_t& gt)
{
    cv::Mat rot1, rot2, dRod;
    cv::Rodrigues(est.first, rot1, dRod);
    cv::Rodrigues(gt.first, rot2);

    // measure loss of inverted poses (camera pose instead of scene pose)
    cv::Mat_<double> invRot1 = rot1.t();
    cv::Mat_<double> invRot2 = rot2.t();

    // get the difference rotation
    cv::Mat diffRot = rot1 * invRot2;

    // calculate rotational and translational error
    double trace = cv::trace(diffRot)[0];
    trace = std::min(3.0, std::max(-1.0, trace));
    double rotErr = 180*acos((trace-1.0)/2.0)/CV_PI;

    cv::Mat_<double> invT1 = est.second * 100;
    invT1 = invRot1 * invT1;

    cv::Mat_<double> invT2 = gt.second * 100;
    invT2 = invRot2 * invT2;

    // zero error, abort
    double tErr = cv::norm(invT1 - invT2);

    cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, 6);

    // clamped loss, return zero gradient if loss is bigger than threshold
    if(std::max(rotErr, tErr) > MAXLOSS)
        return jacobean;

    if((tErr + rotErr) < EPS)
        return jacobean;

    if(tErr > rotErr)
    {
        // return gradient of translational error
        cv::Mat_<double> dDist_dInvT1(1, 3);
        for(unsigned i = 0; i < 3; i++)
            dDist_dInvT1(0, i) = (invT1(i, 0) - invT2(i, 0)) / tErr;

        cv::Mat_<double> dInvT1_dEstT(3, 3);
        dInvT1_dEstT = invRot1 * 100;

        cv::Mat_<double> dDist_dEstT = dDist_dInvT1 * dInvT1_dEstT;
        dDist_dEstT.copyTo(jacobean.colRange(3, 6));

        cv::Mat_<double> dInvT1_dInvRot1 = cv::Mat_<double>::zeros(3, 9);

        dInvT1_dInvRot1(0, 0) = est.second.at<double>(0, 0) * 100;
        dInvT1_dInvRot1(0, 3) = est.second.at<double>(1, 0) * 100;
        dInvT1_dInvRot1(0, 6) = est.second.at<double>(2, 0) * 100;

        dInvT1_dInvRot1(1, 1) = est.second.at<double>(0, 0) * 100;
        dInvT1_dInvRot1(1, 4) = est.second.at<double>(1, 0) * 100;
        dInvT1_dInvRot1(1, 7) = est.second.at<double>(2, 0) * 100;

        dInvT1_dInvRot1(2, 2) = est.second.at<double>(0, 0) * 100;
        dInvT1_dInvRot1(2, 5) = est.second.at<double>(1, 0) * 100;
        dInvT1_dInvRot1(2, 8) = est.second.at<double>(2, 0) * 100;

        dRod = dRod.t();

        cv::Mat_<double> dDist_dRod = dDist_dInvT1 * dInvT1_dInvRot1 * dRod;
        dDist_dRod.copyTo(jacobean.colRange(0, 3));
    }
    else
    {
        // return gradient of rotational error
        dRod = dRod.t();

        cv::Mat_<double> dRotDiff = cv::Mat_<double>::zeros(9, 9);
        invRot2.row(0).copyTo(dRotDiff.row(0).colRange(0, 3));
        invRot2.row(1).copyTo(dRotDiff.row(1).colRange(0, 3));
        invRot2.row(2).copyTo(dRotDiff.row(2).colRange(0, 3));

        invRot2.row(0).copyTo(dRotDiff.row(3).colRange(3, 6));
        invRot2.row(1).copyTo(dRotDiff.row(4).colRange(3, 6));
        invRot2.row(2).copyTo(dRotDiff.row(5).colRange(3, 6));

        invRot2.row(0).copyTo(dRotDiff.row(6).colRange(6, 9));
        invRot2.row(1).copyTo(dRotDiff.row(7).colRange(6, 9));
        invRot2.row(2).copyTo(dRotDiff.row(8).colRange(6, 9));

        dRotDiff = dRotDiff.t();

        cv::Mat_<double> dTrace = cv::Mat_<double>::zeros(1, 9);
        dTrace(0, 0) = 1;
        dTrace(0, 4) = 1;
        dTrace(0, 8) = 1;

        cv::Mat_<double> dAngle = (180 / CV_PI * -1 / sqrt(3 - trace * trace + 2 * trace)) * dTrace * dRotDiff * dRod;
        dAngle.copyTo(jacobean.colRange(0, 3));
    }

    if(cv::sum(cv::Mat(jacobean != jacobean))[0] > 0) //check for NaNs
        return cv::Mat_<double>::zeros(1, 6);

    return jacobean;
}


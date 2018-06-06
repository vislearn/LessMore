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

#include "properties.h"
#include "util.h"
#include "read_data.h"
#include <stdexcept>

namespace jp
{
    /**
     * @brief Calculate the camera coordinate given a pixel position and a depth value.
     *
     * @param x X component of the pixel position.
     * @param y Y component of the pixel position.
     * @param depth Depth value at that position in mm.
     * @return jp::coord3_t Camera coordinate.
     */
    inline cv::Mat_<double> pxToEye(double x, double y, double depth)
    {
        cv::Mat_<double> eye = cv::Mat_<double>::zeros(3, 1);

        if(depth == 0)
            return eye;

        GlobalProperties* gp = GlobalProperties::getInstance();

        eye(0, 0) = ((x - (gp->dP.imageWidth / 2.0 + gp->dP.xShift)) / (gp->dP.focalLength / depth));
        eye(1, 0) = ((y - (gp->dP.imageHeight / 2.0 + gp->dP.yShift)) / (gp->dP.focalLength / depth));
        eye(2, 0) = (jp::coord1_t) depth;

        return eye;
    }

    /**
     * @brief Class that is a interface for reading and writing object specific data.
     *
     */
    class Dataset
    {
        public:

        Dataset()
        {
        }
      
        /**
         * @brief Constructor.
         *
         * @param basePath The directory with subdirectories "rgb", "depth" and "poses".
         */
        Dataset(const std::string& basePath)
	{
            readFileNames(basePath);
	}

        /**
         * @brief Size of the dataset (number of frames).
         *
         * @return size_t Size.
         */
	size_t size() const 
	{ 
            return bgrFiles.size();
	}

        /**
         * @brief Return the RGB image file name of the given frame number.
         *
         * @param i Frame number.
         * @return std::string File name.
         */
	std::string getFileName(size_t i) const
	{
            return bgrFiles[i];
	}
	
        /**
         * @brief Get ground truth pose for the given frame.
         *
         * @param i Frame number.
         * @return bool Returns false if there is no valid ground truth for this frame.
         */
        bool getPose(size_t i, jp::cv_trans_t& pose) const
	{
            if(infoFiles.empty())
                return false;
            if(!readData(infoFiles[i], pose))
                return false;
	    return true;
	}	
	
        /**
         * @brief Get the RGB image of the given frame.
         *
         * @param i Frame number.
         * @param img Output parameter. RGB image.
         * @return void
         */
        void getBGR(size_t i, jp::img_bgr_t& img) const
	{
            std::string bgrFile = bgrFiles[i];
	    
	    readData(bgrFile, img);

            GlobalProperties* gp = GlobalProperties::getInstance();

            int imgWidth = gp->dP.imageWidth;
            int imgHeight = gp->dP.imageHeight;
            int imgPadding = gp->dP.imgPadding;

            // add zero padding to image for random shifting in training mode
            int realImgWidth = gp->getCNNInputDimX();
            int realImgHeight = gp->getCNNInputDimY();

            // rescale input image
            if((img.cols != realImgWidth) || (img.rows != realImgHeight))
            cv::resize(img, img, cv::Size(realImgWidth, realImgHeight));

            jp::img_bgr_t imgPadded = jp::img_bgr_t::zeros(imgHeight, imgWidth);
            img.copyTo(imgPadded.colRange(imgPadding, imgPadding + img.cols).rowRange(imgPadding, imgPadding + img.rows));
            img = imgPadded;
	}

        /**
         * @brief Get the depth image of the given frame.
         *
         * If RGB and Depth are not registered (rawData flag in properties.h), Depth will be
         * mapped to RGB using calibration parameters and the external sensor transformation matrix.
         * If the constD paramters has a positive value, the depth channel will be filled with this value.
         *
         * @param i Frame number.
         * @param img Output parameter. depth image.
         * @return void
         */
        void getDepth(size_t i, jp::img_depth_t& img) const
	{
            if(GlobalProperties::getInstance()->dP.constD > 0)
            {
                // return constant depth channel
                img = jp::img_depth_t::ones(
                    GlobalProperties::getInstance()->dP.imageHeight,
                    GlobalProperties::getInstance()->dP.imageWidth);

                img *= GlobalProperties::getInstance()->dP.constD * 1000;
            }
            else
            {
                std::string dFile = depthFiles[i];

                readData(dFile, img);

                // zero pad image for random shifting in training mode
                int imgPadding = GlobalProperties::getInstance()->dP.imgPadding;
                jp::img_depth_t imgPadded = jp::img_depth_t::zeros(img.rows + 2 * imgPadding, img.cols + 2 * imgPadding);
                img.copyTo(imgPadded.colRange(imgPadding, imgPadding + img.cols).rowRange(imgPadding, imgPadding + img.rows));
                img = imgPadded;
            }
	}

        /**
         * @brief Get the RGB-D image of the given frame.
         *
         * @param i Frame number.
         * @param img Output parameter. RGB-D image.
         * @return void
         */
        void getBGRD(size_t i, jp::img_bgrd_t& img) const
	{
            getBGR(i, img.bgr);
            getDepth(i, img.depth);
	}

        /**
         * @brief Get the ground truth object coordinate image of the given frame.
         *
         * Object coordinates will be generated from image depth and the ground truth pose.
         *
         * @param i Frame number.
         * @param img Output parameter. Object coordinate image.
         * @return void
         */
	void getObj(size_t i, jp::img_coord_t& img) const
	{
            jp::img_depth_t depthData;
            getDepth(i, depthData);

            // get ground truth pose
            jp::cv_trans_t poseData;
            getPose(i, poseData);

            cv::Mat rot;
            cv::Rodrigues(poseData.first, rot);

            img = jp::img_coord_t(depthData.rows, depthData.cols);

            #pragma omp parallel for
            for(unsigned x = 0; x < img.cols; x++)
            for(unsigned y = 0; y < img.rows; y++)
            {
                if(depthData(y, x) == 0)
                {
                    img(y, x) = jp::coord3_t(0, 0, 0);
                    continue;
                }

                // transform depth to camera coordinate
                cv::Mat_<double> eye = pxToEye(x, y, depthData(y, x) / 1000.0);

                // transform camera coordinte to object coordinate
                eye = eye - poseData.second;
                eye = rot.t() * eye;

                img(y, x) = jp::coord3_t(eye(0, 0), eye(1, 0), eye(2, 0));
            }
	}

        private:

        /**
         * @brief Reads all file names in the various sub-folders of a dataset.
         *
         * @param basePath Folder where all data sub folders lie.
         * @return void
         */
	void readFileNames(const std::string& basePath)
	{
            std::cout << "Reading file names... " << std::endl;
            std::string bgrPath = "/rgb/", bgrSuf = ".png";
            std::string dPath = "/depth/", dSuf = ".png";
            std::string infoPath = "/poses/", infoSuf = ".txt";

            bgrFiles = getFiles(basePath + bgrPath, bgrSuf, true);
            if(bgrFiles.empty())
                bgrFiles = getFiles(basePath + bgrPath, ".jpg");
                depthFiles = getFiles(basePath + dPath, dSuf);
            if(depthFiles.empty())
                depthFiles = getFiles(basePath + dPath, ".tiff");
            infoFiles = getFiles(basePath + infoPath, infoSuf, true);

            // optional subsampling of images
            std::vector<std::string> bgrFilesTemp;
            std::vector<std::string> depthFilesTemp;
            std::vector<std::string> infoFilesTemp;

            for(unsigned i = 0; i < bgrFiles.size(); i+=GlobalProperties::getInstance()->dP.imageSubSample)
            {
                if(!bgrFiles.empty()) bgrFilesTemp.push_back(bgrFiles[i]);
                if(!depthFiles.empty()) depthFilesTemp.push_back(depthFiles[i]);
                if(!infoFiles.empty()) infoFilesTemp.push_back(infoFiles[i]);
            }

            bgrFiles = bgrFilesTemp;
            depthFiles = depthFilesTemp;
            infoFiles = infoFilesTemp;
	}
	
        // image data files
	std::vector<std::string> bgrFiles;
        std::vector<std::string> depthFiles;
        // groundtruth data files
        std::vector<std::string> infoFiles;
    };
}

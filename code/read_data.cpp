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


#include "read_data.h"
#include "util.h"

#include <fstream>
#include "png++/png.hpp"

namespace jp
{
    void readData(const std::string dFile, jp::img_depth_t& image)
    {
        if(endsWith(dFile, "png"))
        {
            png::image<unsigned short> imgPng(dFile);
            image = jp::img_depth_t(imgPng.get_height(), imgPng.get_width());

            for(int x = 0; x < imgPng.get_width(); x++)
            for(int y = 0; y < imgPng.get_height(); y++)
            {
                image(y, x) = (jp::depth_t) imgPng.get_pixel(x, y);
            }
        }
        else if(endsWith(dFile, "tiff"))
        {
            image = cv::imread(dFile, CV_LOAD_IMAGE_UNCHANGED);
        }
        else
        {
            std::cout << REDTEXT("ERROR: Unknown file format while reading depth files!") << std::endl;
        }
    }

    void readData(const std::string bgrFile, jp::img_bgr_t& image)
    {
        image = cv::imread(bgrFile);
    }
  
    void readData(const std::string bgrFile, const std::string dFile, jp::img_bgrd_t& image)
    {
        readData(bgrFile, image.bgr);
        readData(dFile, image.depth);
    }

    void readData(const std::string coordFile, jp::img_coord_t& image)
    {
        png::image<png::basic_rgb_pixel<unsigned short>> imgPng(coordFile);
        image = jp::img_coord_t(imgPng.get_height(), imgPng.get_width());

        for(int x = 0; x < imgPng.get_width(); x++)
        for(int y = 0; y < imgPng.get_height(); y++)
        {
            image(y, x)(0) = (jp::coord1_t) imgPng.get_pixel(x, y).red;
            image(y, x)(1) = (jp::coord1_t) imgPng.get_pixel(x, y).green;
            image(y, x)(2) = (jp::coord1_t) imgPng.get_pixel(x, y).blue;
        }
    }


    bool readData(const std::string infoFile, jp::cv_trans_t& pose)
    {
        std::ifstream file(infoFile);
        if(!file.is_open())
        {
            return false;
        }

        std::string line;
        std::vector<std::string> tokens;

        cv::Mat_<double> trans = cv::Mat_<double>::eye(4, 4);

        for(unsigned i = 0; i < 3; i++)
        {
            std::getline(file, line);
            tokens = split(line);

            trans(i, 0) = std::atof(tokens[0].c_str());
            trans(i, 1) = std::atof(tokens[1].c_str());
            trans(i, 2) = std::atof(tokens[2].c_str());
            trans(i, 3) = std::atof(tokens[3].c_str());
        }

        // our code estimates the inverted pose (i.e. the scene pose instead of the camera pose)
        trans = trans.inv();

        // copy rotation
        cv::Rodrigues(trans.colRange(0, 3).rowRange(0, 3), pose.first);

        // copy translation
        pose.second = cv::Mat_<double>(3, 1);
        trans.col(3).rowRange(0, 3).copyTo(pose.second);

        file.close();
        return true;
    }
}



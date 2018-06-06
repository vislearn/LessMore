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


#include "properties.h"
#include "util.h"
#include "thread_rand.h"

#include <iostream>
#include <fstream>
#include <valarray>
#include "generic_io.h"

GlobalProperties* GlobalProperties::instance = NULL;

GlobalProperties::GlobalProperties()
{
    // testing parameters
    tP.sessionString = "";

    tP.objScript = "train_obj.lua";
    tP.scoreScript = "score_incount_ec6.lua";
    tP.objModel = "obj_model_fcn_init.net";

    tP.ransacIterations = 256;
    tP.ransacRefinementIterations = 100;
    tP.ransacInlierThreshold = 10;

    tP.randomDraw = true;

    //dataset parameters
    dP.config = "default";

    dP.focalLength = 585.0f;
    dP.xShift = 0.f;
    dP.yShift = 0.f;
    dP.imgPadding = 4;

    dP.imageWidth = 640;
    dP.imageHeight = 480;
    
    dP.constD = 0;

    dP.imageSubSample = 1;

    dP.cnnSubSample = 8;
}

GlobalProperties* GlobalProperties::getInstance()
{
    if(instance == NULL)
    instance = new GlobalProperties();
    return instance;
}

bool GlobalProperties::readArguments(std::vector<std::string> argv)
{
    int argc = argv.size();

    for(int i = 0; i < argc; i++)
    {
        std::string s = argv[i];

        if(s == "-iw")
        {
            i++;
            dP.imageWidth = std::atoi(argv[i].c_str());
            std::cout << "image width: " << dP.imageWidth << "\n";
            continue;
        }

        if(s == "-ih")
        {
            i++;
            dP.imageHeight = std::atoi(argv[i].c_str());
            std::cout << "image height: " << dP.imageHeight << "\n";
            continue;
        }

        if(s == "-fl")
        {
            i++;
            dP.focalLength = (float)std::atof(argv[i].c_str());
            std::cout << "focal length: " << dP.focalLength << "\n";
            continue;
        }

        if(s == "-xs")
        {
            i++;
            dP.xShift = (float)std::atof(argv[i].c_str());
            std::cout << "x shift: " << dP.xShift << "\n";
            continue;
        }

        if(s == "-ys")
        {
            i++;
            dP.yShift = (float)std::atof(argv[i].c_str());
            std::cout << "y shift: " << dP.yShift << "\n";
            continue;
        }

        if(s == "-cd")
        {
            i++;
            dP.constD = std::atof(argv[i].c_str());
            std::cout << "const depth: " << dP.constD << "\n";
            continue;
        }


        if(s == "-rdraw")
        {
            i++;
            tP.randomDraw = std::atoi(argv[i].c_str());
            std::cout << "random draw: " << tP.randomDraw << "\n";
            continue;
        }

        if(s == "-sid")
        {
            i++;
            tP.sessionString = argv[i];
            std::cout << "session string: " << tP.sessionString << "\n";
            dP.config = tP.sessionString;
            parseConfig();
            continue;
        }

        if(s == "-oscript")
        {
            i++;
            tP.objScript = argv[i];
            std::cout << "object script: " << tP.objScript << "\n";
            continue;
        }

        if(s == "-sscript")
        {
            i++;
            tP.scoreScript = argv[i];
            std::cout << "score script: " << tP.scoreScript << "\n";
            continue;
        }

        if(s == "-omodel")
        {
            i++;
            tP.objModel = argv[i];
            std::cout << "object model: " << tP.objModel << "\n";
            continue;
        }


        if(s == "-rI")
        {
            i++;
            tP.ransacIterations = std::atoi(argv[i].c_str());
            std::cout << "ransac iterations: " << tP.ransacIterations << "\n";
            continue;
        }

        if(s == "-rT")
        {
            i++;
            tP.ransacInlierThreshold = (float)std::atof(argv[i].c_str());
            std::cout << "ransac inlier threshold: " << tP.ransacInlierThreshold << "\n";
            continue;
        }

        if(s == "-rRI")
        {
            i++;
            tP.ransacRefinementIterations = std::atoi(argv[i].c_str());
            std::cout << "ransac iterations (refinement): " << tP.ransacRefinementIterations << "\n";
            continue;
        }

        if(s == "-iSS")
        {
            i++;
            dP.imageSubSample = std::atoi(argv[i].c_str());
            std::cout << "test image sub sampling: " << dP.imageSubSample << "\n";
            continue;
        }

        if(s == "-cSS")
        {
            i++;
            dP.cnnSubSample = std::atoi(argv[i].c_str());
            std::cout << "CNN sub sampling: " << dP.cnnSubSample << "\n";
            continue;
        }

        //nothing found
        {
            std::cout << "unkown argument: " << argv[i] << "\n";
            continue;
        }
    }
}
  
void GlobalProperties::parseCmdLine(int argc, const char* argv[])
{
    std::vector<std::string> argVec;
    for(int i = 1; i < argc; i++) argVec.push_back(argv[i]);
    readArguments(argVec);

    dP.imageWidth += 2 * dP.imgPadding;
    dP.imageHeight += 2 * dP.imgPadding;
}

void GlobalProperties::parseConfig()
{
    std::string configFile = dP.config + ".config";
    std::cout << BLUETEXT("Parsing config file: ") << configFile << std::endl;
    
    std::ifstream file(configFile);
    if(!file.is_open()) return;

    std::vector<std::string> argVec;

    std::string line;
    std::vector<std::string> tokens;
	
    while(true)
    {
        if(file.eof()) break;

        std::getline(file, line);
        if(line.length() == 0) continue;
        if(line.at(0) == '#') continue;

        tokens = split(line);
        if(tokens.empty()) continue;

        argVec.push_back("-" + tokens[0]);
        argVec.push_back(tokens[1]);
    }
    
    readArguments(argVec);
}

cv::Mat_<float> GlobalProperties::getCamMat()
{
    float centerX = dP.imageWidth / 2 + dP.xShift;
    float centerY = dP.imageHeight / 2 + dP.yShift;
    float f = dP.focalLength;

    cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
    camMat(0, 0) = f;
    camMat(1, 1) = f;
    camMat(2, 2) = 1.f;
    
    camMat(0, 2) = centerX;
    camMat(1, 2) = centerY;
    
    return camMat;
}

static GlobalProperties* instance;

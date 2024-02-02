/*
https://github.com/Shiaoming/ALIKE-cpp
BSD 3-Clause License

Copyright (c) 2022, Zhao Xiaoming
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <string>

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <simple_padder.hpp>
#include <soft_detect.hpp>
#include <utils.h>

namespace alike
{
    class ALIKE
    {
    public:
        ALIKE(std::string model_path,
              bool cuda,
              int radius,
              int top_k,
              float scores_th,
              int n_limit,
              bool subpixel);

        void extract(torch::Tensor &img_tensor,
                     torch::Tensor &score_map,
                     torch::Tensor &descriptor_map);
        

        void detect(torch::Tensor &score_map,
                    torch::Tensor &keypoints,
                    torch::Tensor &dispersitys,
                    torch::Tensor &kptscores);

        void compute(torch::Tensor &descriptor_map,
                     torch::Tensor &keypoints,
                     torch::Tensor &descriptors);
        

        void detectAndCompute(torch::Tensor &score_map,
                              torch::Tensor &descriptor_map,
                              torch::Tensor &keypoints,
                              torch::Tensor &dispersitys,
                              torch::Tensor &kptscores,
                              torch::Tensor &descriptors);
        

        void extactAndDetectAndCompute(torch::Tensor &img_tensor,
                                       torch::Tensor &keypoints,
                                       torch::Tensor &dispersitys,
                                       torch::Tensor &kptscores,
                                       torch::Tensor &descriptors);
        

        void toOpenCVFormat(torch::Tensor &keypoints_t,
                            torch::Tensor &dispersitys_t,
                            torch::Tensor &kptscores_t,
                            torch::Tensor &descriptors_t,
                            std::vector<cv::KeyPoint> &keypoints, // x,y; size: dispersity; response: score
                            cv::Mat &descriptors);
    

    private:
        DKD mDkd;
        bool mSubPixel;
        torch::jit::script::Module mJitModel;
        torch::Device mDevice;
    };
}
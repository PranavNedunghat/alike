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
#include <chrono>
#include <torch/torch.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/opencv.hpp>
#include <image_loader.hpp>
#include <alike.hpp>
#include <simple_tracker.hpp>
#include "sensor_msgs/CompressedImage.h"
#include <utils.h>
#include <iostream>
#include <sstream>
#include <read_config.h>

using std::stringstream;

using namespace alike;

class Config
{
    public:
    std::string config_path;
    std::string model_path;
    bool use_cuda;
    int top_k;
    float scores_th;
    int n_limit;
    int max_size;
    float ratio;
    bool no_display;
    bool no_subpixel;

    Config()
    {
        config_path = "/home/pranav/catkin_ws/src/alike/config/config.yaml";
        Configs config(config_path);
        model_path = config.model_dir;
        use_cuda = config.alike_config.use_cuda;
        top_k = config.alike_config.top_k;
        scores_th = config.alike_config.scores_threshold;
        n_limit = config.alike_config.max_keypoints;
        max_size = config.alike_config.max_image_size;
        ratio = config.alike_config.ratio;
        no_display = config.alike_config.no_display;
        no_subpixel = config.alike_config.no_subpixel;

    }
};

class ALIKE_Tracker
{
    public:
    ros::NodeHandle node;
	ros::Subscriber sub;
	std::shared_ptr<sensor_msgs::CompressedImage> image1;
    Config record;
    ALIKE alike;
    SimpleTracker tracker = SimpleTracker();
    torch::Device device;
    std::vector<int> runtimes;
    ALIKE_Tracker(Config config):alike(config.model_path,config.use_cuda,2,config.top_k,config.scores_th,config.n_limit,!config.no_subpixel), device((config.use_cuda) ? torch::kCUDA : torch::kCPU)
    {
        record = config;
        std::cout << "=======================" << std::endl;
        std::cout << "Running with " << ((config.use_cuda) ? "CUDA" : "CPU") << "!" << std::endl;
        std::cout << "=======================" << std::endl;
        node = ros::NodeHandle("alike_node");
        sub = node.subscribe("/race12/cam1/color/image_raw/compressed",1,&ALIKE_Tracker::matches_callback,this);
    }

    void matches_callback(const sensor_msgs::CompressedImage::ConstPtr& msg)
    {
        image1 = std::make_shared<sensor_msgs::CompressedImage>(*msg);
		if (image1)
		{	
			cv::Mat image(cv::imdecode(image1->data, cv::IMREAD_ANYCOLOR));
            torch::Tensor score_map, descriptor_map;
            torch::Tensor keypoints_t, dispersitys_t, kptscores_t, descriptors_t;
            std::vector<cv::KeyPoint>keypoints;
            cv::Mat descriptors;
            cv::Mat img_rgb = image;
            auto img_tensor = mat2Tensor(image).permute({2, 0, 1}).unsqueeze(0).to(device).to(torch::kFloat) / 255;

            // core
            using namespace std::chrono;
            if (record.use_cuda)
                torch::cuda::synchronize();
            auto start = system_clock::now();
            // ====== core
            alike.extract(img_tensor, score_map, descriptor_map);
            alike.detectAndCompute(score_map, descriptor_map, keypoints_t, dispersitys_t, kptscores_t, descriptors_t);
            // ====== core
            if (record.use_cuda)
                torch::cuda::synchronize();
            auto end = system_clock::now();
            milliseconds mill = duration_cast<milliseconds>(end - start);
            runtimes.push_back(mill.count());

            if (keypoints_t.numel() > 2)
            {
                // Note: for a keypoint of cv::KeyPoint
                // keypoint.size=dispersity is the dispersity
                // keypoint.response=score is the score
                alike.toOpenCVFormat(keypoints_t, dispersitys_t, kptscores_t, descriptors_t, keypoints, descriptors);
                cv::Mat track_img = image.clone();
                auto N_matches = tracker.update(track_img, keypoints, descriptors);

                // get fps
                float ave_fps = 0;
                for (auto i = 0; i < runtimes.size(); i++)
                    ave_fps += 1000 / runtimes[i];
                ave_fps = ave_fps / runtimes.size();
                stringstream fmt;
                fmt << "FPS: " << ave_fps << ", Keypoints/Matches: " << keypoints.size() << "/" << N_matches;
                std::string status = fmt.str();
                //std::cout << status << std::endl;

                // visualization
                if (!record.no_display)
                {
                    cv::setWindowTitle("win", status);
                    cv::imshow("win", track_img);
                    cv::waitKey(1);
                }
            }
            else
            {
                std::cout << "No Keypoints detected!" << std::endl;
            }
            


		}
    }
};


int main(int argc, char** argv)
{

    ros::init(argc, argv, "alike_node");
    Config config;
	ALIKE_Tracker alike_node(config);
	ros::spin();
    std::cout << "Finished!" << std::endl;
    
    return 0;
}
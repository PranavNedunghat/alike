#ifndef READ_CONFIG_H_
#define READ_CONFIG_H_

#include "utils.h"
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace alike
{
struct AlikeConfig {
  int max_keypoints{};
  bool use_cuda;
  int top_k{};
  float scores_threshold{};
  float ratio{};
  int max_image_size{};
  bool no_display;
  bool no_subpixel;
  std::string model_path;
  std::string model_name;
};




struct Configs{
  std::string model_dir;

  AlikeConfig alike_config;

  Configs(const std::string& config_file){
    std::cout << "Config file is " << config_file << std::endl;
    if(!alike::isFileExist(config_file)){
      std::cerr << "Config file " << config_file << " doesn't exist." << std::endl;
      return;
    }
    YAML::Node file_node = YAML::LoadFile(config_file);

    YAML::Node alike_node = file_node["alike"];
    alike_config.max_keypoints = alike_node["max_keypoints"].as<int>();
    alike_config.top_k = alike_node["top_k"].as<int>();
    alike_config.scores_threshold = alike_node["scores_threshold"].as<float>();
    alike_config.ratio = alike_node["ratio"].as<float>();
    alike_config.max_image_size = alike_node["max_image_size"].as<int>();
    alike_config.use_cuda = alike_node["use_cuda"].as<bool>();
    alike_config.no_display = alike_node["no_display"].as<bool>();
    alike_config.no_subpixel = alike_node["no_subpixel"].as<bool>();
    alike_config.model_path = alike_node["model_path"].as<std::string>();
    alike_config.model_name = alike_node["model_name"].as<std::string>();
    model_dir = alike_config.model_path + "/" + alike_config.model_name;
  }
};
}
#endif  // READ_CONFIG_H_

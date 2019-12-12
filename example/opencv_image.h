/**************************************************************************
* VIG-Init
*
* Copyright SenseTime. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
**************************************************************************/

#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <slamtools/common.h>

class OpenCvImage : public Image {
  public:
    OpenCvImage();
    OpenCvImage(const std::string &filename);
    void detect_keypoints(std::vector<Eigen::Vector2d> &keypoints, size_t max_points = 0) const override;
    void track_keypoints(const Image *next_image, const std::vector<Eigen::Vector2d> &curr_keypoints, std::vector<Eigen::Vector2d> &next_keypoints, std::vector<char> &result_status) const override;
    void detect_segments(std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d>> &segments, size_t max_segments = 0) const override;

    void preprocess();
    void correct_distortion(const Eigen::Matrix3d &intrinsics, const Eigen::Vector4d &coeffs);
    cv::Mat image;

  private:
    static cv::CLAHE *clahe();
    static cv::line_descriptor::LSDDetector *lsd();
    static cv::GFTTDetector *gftt();
    static cv::FastFeatureDetector *fast();
    static cv::ORB *orb();
};

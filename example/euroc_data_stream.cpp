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

#include "euroc_data_stream.h"
#include <regex>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <slamtools/configurator.h>
#include "opencv_image.h"

using namespace Eigen;

struct EurocDataConfig : public Configurator {
    EurocDataConfig(const std::string &camera_yaml_path, const std::string &imu_yaml_path) {
        try {
            YAML::Node camera_node = YAML::LoadFile(camera_yaml_path);

            YAML::Node k_node = camera_node["intrinsics"];
            K.setIdentity();
            K(0, 0) = k_node[0].as<double>();
            K(1, 1) = k_node[1].as<double>();
            K(0, 2) = k_node[2].as<double>();
            K(1, 2) = k_node[3].as<double>();

            YAML::Node cbs_node = camera_node["T_BS"]["data"];
            Matrix3d R_cam2body;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R_cam2body(i, j) = cbs_node[i * 4 + j].as<double>();
                }
                p_cam2body(i) = cbs_node[i * 4 + 3].as<double>();
            }
            q_cam2body = R_cam2body;
        } catch (...) { // for the sake of example, we don't really handle the errors.
            throw;
        }

        try {
            YAML::Node imu_node = YAML::LoadFile(imu_yaml_path);
            YAML::Node ibs_node = imu_node["T_BS"]["data"];
            Matrix3d R_imu2body;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R_imu2body(i, j) = ibs_node[i * 4 + j].as<double>();
                }
                p_imu2body(i) = ibs_node[i * 4 + 3].as<double>();
            }
            q_imu2body = R_imu2body;

            const double sigma_correction = 1.0;
            double sigma_w = imu_node["gyroscope_noise_density"].as<double>() * sigma_correction;
            double sigma_a = imu_node["accelerometer_noise_density"].as<double>() * sigma_correction;
            double sigma_bg = imu_node["gyroscope_random_walk"].as<double>() * sigma_correction;
            double sigma_ba = imu_node["accelerometer_random_walk"].as<double>() * sigma_correction;
            gyro_noise = sigma_w * sigma_w * Matrix3d::Identity();
            accl_noise = sigma_a * sigma_a * Matrix3d::Identity();
            gyro_random_walk = sigma_bg * sigma_bg * Matrix3d::Identity();
            accl_random_walk = sigma_ba * sigma_ba * Matrix3d::Identity();
        } catch (...) {
            throw;
        }
    }

    Matrix3d camera_intrinsic() const override {
        return K;
    }

    Quaterniond camera_to_center_rotation() const override {
        return q_cam2body;
    }

    Vector3d camera_to_center_translation() const override {
        return p_cam2body;
    }

    Quaterniond imu_to_center_rotation() const override {
        return q_imu2body;
    }

    Vector3d imu_to_center_translation() const override {
        return p_imu2body;
    }

    Matrix3d imu_gyro_white_noise() const override {
        return gyro_noise;
    }

    Matrix3d imu_accel_white_noise() const override {
        return accl_noise;
    }

    Matrix3d imu_gyro_random_walk() const override {
        return gyro_random_walk;
    }

    Matrix3d imu_accel_random_walk() const override {
        return accl_random_walk;
    }

  private:
    Matrix3d K;
    Quaterniond q_cam2body;
    Vector3d p_cam2body;
    Quaterniond q_imu2body;
    Vector3d p_imu2body;
    Matrix3d gyro_noise;
    Matrix3d gyro_random_walk;
    Matrix3d accl_noise;
    Matrix3d accl_random_walk;
};

EurocDataStream::EurocDataStream(const std::string &path) {
    std::string cam0_yaml_path = path + "/mav0/cam0/sensor.yaml";
    data_config = std::make_shared<EurocDataConfig>(cam0_yaml_path, path + "/mav0/imu0/sensor.yaml");
    YAML::Node cam0_node = YAML::LoadFile(cam0_yaml_path);
    YAML::Node dc_node = cam0_node["distortion_coefficients"];
    distortion_coeffs[0] = dc_node[0].as<double>();
    distortion_coeffs[1] = dc_node[1].as<double>();
    distortion_coeffs[2] = dc_node[2].as<double>();
    distortion_coeffs[3] = dc_node[3].as<double>();

    std::string camera_filename = path + "/mav0/cam0/data.csv";

    std::ifstream camera_file(camera_filename.c_str(), std::ifstream::in);
    std::regex camera_data_regex("^([[:digit:]]*),([^[:space:]]*)[[:space:]]*$", std::regex::ECMAScript);
    std::string camera_header_line;
    std::getline(camera_file, camera_header_line);
    std::string camera_data_line;
    while (std::getline(camera_file, camera_data_line)) {
        std::smatch matches;
        std::regex_search(camera_data_line, matches, camera_data_regex);
        image_list.emplace_back(double(std::stoull(matches[1])) * double(1.0e-9), path + "/mav0/cam0/data/" + std::string(matches[2]));
    }

    std::string imu_filename = path + "/mav0/imu0/data.csv";
    std::ifstream imu_file(imu_filename.c_str(), std::ifstream::in);
    std::regex imu_data_regex("^([[:digit:]]*),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+)[[:space:]]*$", std::regex::ECMAScript);
    std::string imu_header_line;
    std::getline(imu_file, imu_header_line);
    std::string imu_data_line;
    while (std::getline(imu_file, imu_data_line)) {
        std::smatch matches;
        std::regex_search(imu_data_line, matches, imu_data_regex);
        IMUData data;
        data.t = double(std::stoull(matches[1])) * double(1.0e-9);
        data.w.x() = std::stod(matches[2]);
        data.w.y() = std::stod(matches[3]);
        data.w.z() = std::stod(matches[4]);
        data.a.x() = std::stod(matches[5]);
        data.a.y() = std::stod(matches[6]);
        data.a.z() = std::stod(matches[7]);
        imu_list.push_back(data);
    }

    std::string gt_filename = path + "/mav0/state_groundtruth_estimate0/data.csv";
    std::ifstream gt_file(gt_filename.c_str(), std::ifstream::in);
    std::regex gt_data_regex("^([[:digit:]]*),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+),(-?[[:digit:]]*\\.?[[:digit:]]+)[[:space:]]*$", std::regex::ECMAScript);
    std::string gt_header_line;
    std::getline(gt_file, gt_header_line);
    std::string gt_data_line;
    while (std::getline(gt_file, gt_data_line)) {
        std::smatch matches;
        std::regex_search(gt_data_line, matches, gt_data_regex);
        double t = double(std::stoull(matches[1])) * double(1.0e-9);
        PoseState pose;
        pose.p.x() = std::stod(matches[2]);
        pose.p.y() = std::stod(matches[3]);
        pose.p.z() = std::stod(matches[4]);
        pose.q.w() = std::stod(matches[5]);
        pose.q.x() = std::stod(matches[6]);
        pose.q.y() = std::stod(matches[7]);
        pose.q.z() = std::stod(matches[8]);
        gtpose_list[t] = pose;
    }
}

std::shared_ptr<Configurator> EurocDataStream::config() const {
    return data_config;
}

DataType EurocDataStream::next() {
    if (image_list.size() == 0 && imu_list.size() == 0) {
        return DT_END;
    }
    double t_image = std::numeric_limits<double>::max(), t_imu = std::numeric_limits<double>::max();
    if (image_list.size() > 0) {
        t_image = image_list.front().first;
    }
    if (imu_list.size() > 0) {
        t_imu = imu_list.front().t;
    }
    if (t_imu < t_image) {
        return DT_IMU;
    } else {
        return DT_IMAGE;
    }
}

IMUData EurocDataStream::read_imu() {
    IMUData data = imu_list.front();
    imu_list.pop_front();
    return data;
}

std::shared_ptr<Image> EurocDataStream::read_image() {
    static Vector3d gravity = {0, 0, -GRAVITY_NOMINAL};
    double t = image_list.front().first;
    std::string path = image_list.front().second;
    std::shared_ptr<OpenCvImage> img = std::make_shared<OpenCvImage>(path);
    img->correct_distortion(data_config->camera_intrinsic(), distortion_coeffs);
    img->preprocess();
    img->t = t;
    PoseState pose = ground_truth_pose(t);
    img->g = pose.q.conjugate() * gravity;
    image_list.pop_front();
    return img;
}

bool EurocDataStream::has_ground_truth() const {
    return gtpose_list.size() > 0;
}

PoseState EurocDataStream::ground_truth_pose(double t) const {
    if (gtpose_list.size() == 0) {
        PoseState s;
        s.p.setZero();
        s.q.setIdentity();
        return s;
    } else if (gtpose_list.count(t) > 0) {
        return gtpose_list.at(t);
    } else {
        auto it = gtpose_list.lower_bound(t);
        if (it == gtpose_list.begin()) {
            return it->second;
        } else if (it == gtpose_list.end()) {
            return gtpose_list.rbegin()->second;
        } else {
            auto it0 = std::prev(it);
            double lambda = (t - it0->first) / (it->first - it0->first);
            PoseState ret;
            ret.q = it0->second.q.slerp(lambda, it->second.q);
            ret.p = it0->second.p * (1 - lambda) + it->second.p * lambda;
            return ret;
        }
    }
}

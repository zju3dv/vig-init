/**************************************************************************
* This file is part of vig-init.
* Author: Jinyu Li <mail@jinyu.li>
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

#include <iostream>
#include <fstream>
#include <slamtools/configurator.h>
#include <slamtools/sliding_window.h>
#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <vig_initializer.h>
#include "euroc_data_stream.h"

using namespace Eigen;

std::unique_ptr<Frame> create_frame(std::shared_ptr<Configurator> config) {
    std::unique_ptr<Frame> frame = std::make_unique<Frame>();
    frame->K = config->camera_intrinsic();
    frame->sqrt_inv_cov = frame->K.block<2, 2>(0, 0) / ::sqrt(config->keypoint_pixel_error());
    frame->camera.q_cs = config->camera_to_center_rotation();
    frame->camera.p_cs = config->camera_to_center_translation();
    frame->imu.q_cs = config->imu_to_center_rotation();
    frame->imu.p_cs = config->imu_to_center_translation();
    frame->preintegration.cov_w = config->imu_gyro_white_noise();
    frame->preintegration.cov_a = config->imu_accel_white_noise();
    frame->preintegration.cov_bg = config->imu_gyro_random_walk();
    frame->preintegration.cov_ba = config->imu_accel_random_walk();
    return frame;
}

int main(int argc, char * argv[]) {
    std::unique_ptr<SlidingWindow> initialized_sliding_window; // <-- will put initialized sliding window here

    std::unique_ptr<EurocDataStream> stream = std::make_unique<EurocDataStream>(argv[1]);
    std::unique_ptr<Initializer> initializer = std::make_unique<VigInitializer>(stream->config());

    size_t frame_count = 0;
    // We'll assemble frames, and feed the frames into the initializer
    std::unique_ptr<Frame> pending_frame = create_frame(stream->config());
    for (DataType next = stream->next(); next != DT_END; next = stream->next()) {
        if (next == DT_IMAGE) {
            pending_frame->image = stream->read_image();

            // We also provide algorithms which uses external gravity instead of aliging from vertical edges.
            // You can remove this line if not working with the related algorithms.
            pending_frame->external_gravity = pending_frame->image->g;
            initializer->append_frame(std::move(pending_frame));

            std::cout << "# Frames: " << ++frame_count << std::endl;

            // try initialize, if success, returns the initialized sliding window.
            if(std::unique_ptr<SlidingWindow> sw = initializer->init()) {
                initialized_sliding_window.swap(sw);
                break;
            }

            pending_frame = create_frame(stream->config());
        } else if (next == DT_IMU) {
            pending_frame->preintegration.data.push_back(stream->read_imu());
        }
    }

    // if (initialized_sliding_window) {
    //     the initialized system is oriented such that gravity is pointing to -Z.
    // }

    return 0;
}

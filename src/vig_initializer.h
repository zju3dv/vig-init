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

#include <slamtools/common.h>
#include <slamtools/initializer.h>

class Frame;
class SlidingWindow;
class Configurator;

class VigInitializer : public Initializer {
  public:
    VigInitializer(std::shared_ptr<Configurator> config);
    virtual ~VigInitializer();

    void append_frame(std::unique_ptr<Frame> frame) override;
    std::unique_ptr<SlidingWindow> init() const override;

    bool init_imu(SlidingWindow *map) const;
};

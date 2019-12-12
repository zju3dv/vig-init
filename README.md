# Rapid and Robust Monocular Visual-Inertial Initialization with Gravity Estimation via Vertical Edges

This repository contains the implementation for the corresponding paper.
The main algorithm in the paper are implemented in `vig_initializer.cpp` and `imu_initializer.cpp`.
Welcome feedback.

## How to use

For compilation:

* Install the dependencies: Eigen, Ceres Solver and OpenCV (>= 3.0, < 4.0).
* Clone the repository.
* Populate the submodule with `git submodule init && git submodule update`
* Build with `cmake -B build && cmake --build build`, you will need a compiler supporting C++17.

For integration:

* Inherit `class Configurator` and provide necessary parameters (see example).
* Inherit `class Image` and implement feature extraction, feature matching and line segment detection (see example).
* Assemble `Frame`s and feed into the instance of `VigInitializer` (see example).
* Obtain the initialized `SlidingWindow` from `VigInitializer` (see example).

For customization, we provided different algorithms in `ImuInitializer`, you can tweak the stages of the IMU initialization with them.

## Publication

If you used our work, please kindly cite the following paper.

> **Rapid and Robust Monocular Visual-Inertial Initialization with Gravity Estimation via Vertical Edges**, Jinyu Li, Hujun Bao, and Guofeng Zhang, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS, 2019).

## Caveats

* Precompiled OpenCV 4+ distributions does not have LSD segment detector by default. We used OpenCV 3 in our original experiments.
* In `ImuInitializer`, we provided recipies to emulate the initialization of VINS-Mono and VI-ORB-SLAM. They are only for reference. For reproducing the results in our paper, it is recommended to experiment with the original VINS-Mono and ORB-SLAM.
* For any technical issues, please contact Jinyu Li <mail(AT)jinyu.li>. For commercial inquiries, please contact Guofeng Zhang <zhangguofeng(AT)cad.zju.edu.cn>.

## Acknowledgement

This work is affliated with ZJU-SenseTime Joint Lab of 3D Vision, and its intellectual property belongs to SenseTime Group Ltd.

```
Copyright SenseTime. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

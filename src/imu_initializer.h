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

class SlidingWindow;
class Configurator;

class ImuInitializer {
  public:
    ImuInitializer(SlidingWindow *map, std::shared_ptr<Configurator> config);
    virtual ~ImuInitializer();

    void clear();
    bool init();

  private:
    void integrate();

    void solve_gyro_bias();                               //!1
    void solve_gyro_bias_exhaust();                       //!2
                                                          //
    void solve_gravity_from_external();                   //
                                                          //
    void solve_gravity_scale();                           // solve gravity and scale, assuming zero dba.
    void solve_gravity_scale_exhaust();                   // same as above but exhaustively enumerate all frame combinations.
    void solve_gravity_scale_velocity();                  //! solve gravity scale and velocity assuming zero dba.
    void solve_gravity_scale_velocity_exhaust();          //! same as above but exhaustively enumerate all frame combinations.
                                                          //
    void solve_scale_given_gravity();                     //! assuming zero dba.
    void solve_scale_given_gravity_exhaust();             //
    void solve_scale_ba_given_gravity();                  //
    void solve_scale_ba_given_gravity_exhaust();          //
    void solve_scale_velocity_given_gravity();            // assuming zero dba.
    void solve_scale_velocity_given_gravity_exhaust();    //
    void solve_scale_ba_velocity_given_gravity();         //
    void solve_scale_ba_velocity_given_gravity_exhaust(); //
                                                          //
    void refine_scale_ba_via_gravity();                   //
    void refine_scale_ba_via_gravity_exhaust();           //
    void refine_scale_velocity_via_gravity();             //!1
    void refine_scale_velocity_via_gravity_exhaust();     //!2

    bool refine_gravity_via_visual();
    bool apply_init(bool apply_ba = false, bool apply_velocity = true);

    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    Eigen::Vector3d gravity;
    double scale;
    std::vector<Eigen::Vector3d> velocities;

    SlidingWindow *map;
    std::shared_ptr<Configurator> config;
};

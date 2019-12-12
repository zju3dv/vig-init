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

#pragma once

#include <slamtools/common.h>

class PoissonKeypointFilter {
  public:
    PoissonKeypointFilter(double x_min, double x_max, double y_min, double y_max, double radius) {
        m_x_min = x_min;
        m_x_max = x_max;
        m_y_min = y_min;
        m_y_max = y_max;
        m_radius = radius;
        m_radius_squared = radius * radius;
        m_grid_size = radius / sqrt(2);
        m_grid_width = (int)((m_x_max - m_x_min) / m_grid_size) + 1;
        m_grid_height = (int)((m_y_max - m_y_min) / m_grid_size) + 1;
        clear();
    }

    void clear() {
        std::vector<size_t>(m_grid_width * m_grid_height, nil()).swap(m_grid);
        m_points.clear();
    }

    void set_points(const std::vector<Eigen::Vector2d> &points) {
        for (const Eigen::Vector2d &p : points) {
            int ix, iy;
            to_icoord(p, ix, iy);
            m_grid[ix + iy * m_grid_width] = m_points.size();
            m_points.push_back(p);
        }
    }

    void filter(std::vector<Eigen::Vector2d> &points) {
        size_t n_points_before = m_points.size();
        for (const Eigen::Vector2d &p : points) {
            int ix, iy;
            if (test_point(p, ix, iy)) {
                m_grid[ix + iy * m_grid_width] = m_points.size();
                m_points.push_back(p);
            }
        }
        std::vector<Eigen::Vector2d>(m_points.begin() + n_points_before, m_points.end()).swap(points);
    }

  private:
    void to_icoord(const Eigen::Vector2d &p, int &ix, int &iy) const {
        ix = int(floor((p.x() - m_x_min) / m_grid_size));
        iy = int(floor((p.y() - m_y_min) / m_grid_size));
    }

    bool test_point(const Eigen::Vector2d &p, int &ix, int &iy) const {
        if (p.x() < m_x_min || p.x() > m_x_max) return false;
        if (p.y() < m_y_min || p.y() > m_y_max) return false;
        to_icoord(p, ix, iy);
        int x_extent_begin = std::max(ix - 2, 0), x_extent_end = std::min(ix + 2, m_grid_width - 1);
        int y_extent_begin = std::max(iy - 2, 0), y_extent_end = std::min(iy + 2, m_grid_height - 1);
        for (int y = y_extent_begin; y <= y_extent_end; ++y) {
            for (int x = x_extent_begin; x <= x_extent_end; ++x) {
                size_t nbr = m_grid[x + y * m_grid_width];
                if (nbr != nil()) {
                    if ((p - m_points[nbr]).squaredNorm() < m_radius_squared) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    double m_x_min;
    double m_x_max;
    double m_y_min;
    double m_y_max;
    double m_radius;
    double m_radius_squared;
    double m_grid_size;
    int m_grid_width;
    int m_grid_height;
    std::vector<size_t> m_grid;
    std::vector<Eigen::Vector2d> m_points;
};

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

#include "opencv_image.h"
#include "keypoint_filter.h"

using namespace Eigen;
using namespace cv;

static std::vector<Point2f> to_opencv(const std::vector<Vector2d> &v) {
    std::vector<Point2f> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        r[i].x = v[i].x();
        r[i].y = v[i].y();
    }
    return r;
}

static std::vector<Vector2d> from_opencv(const std::vector<Point2f> &v) {
    std::vector<Vector2d> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        r[i].x() = v[i].x;
        r[i].y() = v[i].y;
    }
    return r;
}

OpenCvImage::OpenCvImage() = default;

OpenCvImage::OpenCvImage(const std::string &filename) {
    image = imread(filename, cv::IMREAD_GRAYSCALE);
}

void OpenCvImage::detect_keypoints(std::vector<Vector2d> &keypoints, size_t max_points) const {
    if (max_points == 0) {
        max_points = 100;
    }

    // std::vector<Point2f> corners;
    // cv::goodFeaturesToTrack(image, corners, max_points, 1.0e-5, 30, cv::noArray(), 5);

    // if (corners.size() > 0) {
    //     std::vector<vector<2>> new_keypoints = from_opencv(corners);

    //     tools::PoissonKeypointFilter filter(20, image.cols - 20, 20, image.rows - 20, 30.0);
    //     filter.set_points(keypoints);
    //     filter.filter(new_keypoints);

    //     keypoints.insert(keypoints.end(), new_keypoints.begin(), new_keypoints.end());
    // }

    std::vector<cv::KeyPoint> cvkeypoints;
    gftt()->detect(image, cvkeypoints);

    if (cvkeypoints.size() > 0) {
        std::sort(cvkeypoints.begin(), cvkeypoints.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
            return a.response > b.response;
        });
        std::vector<Vector2d> new_keypoints;
        for (size_t i = 0; i < cvkeypoints.size(); ++i) {
            new_keypoints.emplace_back(cvkeypoints[i].pt.x, cvkeypoints[i].pt.y);
        }
        PoissonKeypointFilter filter(20, image.cols - 20, 20, image.rows - 20, 20.0);
        filter.set_points(keypoints);
        filter.filter(new_keypoints);

        // std::vector<cv::KeyPoint> filtered_cvkeypoints(new_keypoints.size());
        // for (size_t i = 0; i < new_keypoints.size(); ++i) {
        //     filtered_cvkeypoints[i].pt.x = new_keypoints[i].x();
        //     filtered_cvkeypoints[i].pt.y = new_keypoints[i].y();
        // }
        // cv::Mat cvdescriptors;
        // orb()->compute(image, filtered_cvkeypoints, cvdescriptors);

        keypoints.insert(keypoints.end(), new_keypoints.begin(), new_keypoints.end());
    }
}

void OpenCvImage::track_keypoints(const Image *next_image, const std::vector<Vector2d> &curr_keypoints, std::vector<Vector2d> &next_keypoints, std::vector<char> &result_status) const {
    std::vector<Point2f> curr_cvpoints = to_opencv(curr_keypoints);
    std::vector<Point2f> next_cvpoints;
    if (next_keypoints.size() > 0) {
        next_cvpoints = to_opencv(next_keypoints);
    } else {
        next_keypoints.resize(curr_keypoints.size());
        next_cvpoints = curr_cvpoints;
    }

    const OpenCvImage *next_cvimage = dynamic_cast<const OpenCvImage *>(next_image);

    result_status.resize(curr_keypoints.size(), 0);
    if (next_cvimage && curr_cvpoints.size() > 0) {
        Mat cvstatus, cverr;
        calcOpticalFlowPyrLK(image, next_cvimage->image, curr_cvpoints, next_cvpoints, cvstatus, cverr); //Size(21, 21), 3, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01), OPTFLOW_USE_INITIAL_FLOW);
        for (size_t i = 0; i < next_cvpoints.size(); ++i) {
            result_status[i] = cvstatus.at<unsigned char>(i);
            if (next_cvpoints[i].x < 20 || next_cvpoints[i].x >= image.cols - 20 || next_cvpoints[i].y < 20 || next_cvpoints[i].y >= image.rows - 20) {
                result_status[i] = 0;
            }
        }
    }

    std::vector<size_t> l;
    std::vector<Point2f> p, q;
    for (size_t i = 0; i < result_status.size(); ++i) {
        if (result_status[i] != 0) {
            l.push_back(i);
            p.push_back(curr_cvpoints[i]);
            q.push_back(next_cvpoints[i]);
        }
    }
    if (l.size() >= 8) {
        Mat mask;
        findFundamentalMat(p, q, cv::FM_RANSAC, 1.0, 0.99, mask);
        for (size_t i = 0; i < l.size(); ++i) {
            if (mask.at<unsigned char>(i) == 0) {
                result_status[l[i]] = 0;
            }
        }
        for (size_t i = 0; i < curr_keypoints.size(); ++i) {
            if (result_status[i]) {
                next_keypoints[i].x() = next_cvpoints[i].x;
                next_keypoints[i].y() = next_cvpoints[i].y;
            }
        }
    }
}

void OpenCvImage::detect_segments(std::vector<std::tuple<Vector2d, Vector2d>> &segments, size_t) const {
    std::vector<cv::line_descriptor::KeyLine> keylines;
    lsd()->detect(image, keylines, 2, 2);
    segments.resize(keylines.size());
    for (size_t i = 0; i < keylines.size(); ++i) {
        std::get<0>(segments[i]).x() = keylines[i].startPointX;
        std::get<0>(segments[i]).y() = keylines[i].startPointY;
        std::get<1>(segments[i]).x() = keylines[i].endPointX;
        std::get<1>(segments[i]).y() = keylines[i].endPointY;
    }
}

void OpenCvImage::preprocess() {
    clahe()->apply(image, image);
}

void OpenCvImage::correct_distortion(const Matrix3d &intrinsics, const Vector4d &coeffs) {
    cv::Mat new_image;
    cv::Mat K(3, 3, CV_32FC1), cvcoeffs(1, 4, CV_32FC1);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            K.at<float>(i, j) = intrinsics(i, j);
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        cvcoeffs.at<float>(i) = coeffs(i);
    }
    cv::undistort(image, new_image, K, cvcoeffs);
    image = new_image;
}

cv::CLAHE *OpenCvImage::clahe() {
    static cv::Ptr<cv::CLAHE> s_clahe = cv::createCLAHE(6.0);
    return s_clahe.get();
}

cv::line_descriptor::LSDDetector *OpenCvImage::lsd() {
    static cv::Ptr<cv::line_descriptor::LSDDetector> s_lsd = cv::line_descriptor::LSDDetector::createLSDDetector();
    return s_lsd.get();
}

cv::GFTTDetector *OpenCvImage::gftt() {
    static cv::Ptr<cv::GFTTDetector> s_gftt = cv::GFTTDetector::create(1000, 1.0e-3, 20, 3, true);
    return s_gftt.get();
}

cv::FastFeatureDetector *OpenCvImage::fast() {
    static cv::Ptr<cv::FastFeatureDetector> s_fast = cv::FastFeatureDetector::create();
    return s_fast.get();
}

cv::ORB *OpenCvImage::orb() {
    static cv::Ptr<cv::ORB> s_orb = cv::ORB::create();
    return s_orb.get();
}

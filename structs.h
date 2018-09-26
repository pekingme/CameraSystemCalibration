#ifndef STRUCT_H
#define STRUCT_H

#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/aruco/charuco.hpp"

using namespace std;
using namespace cv;

struct CameraSystemCalibrationOptions {
    double frame_gap; // Gap between frames to check in seconds
    Ptr<aruco::Dictionary> dictionary;
    Ptr<aruco::CharucoBoard> charuco_board;
    Ptr<aruco::DetectorParameters> detector_parameters;
};

struct Frame {
    unsigned global_index;
    string camera_name;
    Mat original_frame;
    unsigned corner_count;
    bool valid;
    Mat corner_ids;
    Mat detected_corners_32; // N x 1 x 32FC2
    Mat board_corners_32; // N x 1 x 32FC3
    Mat flatten_detected_corners_64; // N x 2 x 64FC1
    Mat flatten_board_corners_64; // N x 3 x 64FC1
};

#endif // STRUCT_H

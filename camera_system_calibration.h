#ifndef CAMERA_SYSTEM_CALIBRATION_H
#define CAMERA_SYSTEM_CALIBRATION_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class CameraSystemCalibration
{
public:
    CameraSystemCalibration ( const string& output_file_name, const string& config_file_name, const string& detector_file_name );
    ~CameraSystemCalibration();
private:
    vector<string> GetVideoFileNames ( const string& config_file_name );
};

#endif // CAMERA_SYSTEM_CALIBRATION_H

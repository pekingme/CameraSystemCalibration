#ifndef CAMERA_SYSTEM_CALIBRATION_H
#define CAMERA_SYSTEM_CALIBRATION_H

#include <string>

using namespace std;

class CameraSystemCalibration
{
public:
    CameraSystemCalibration(const string& output_file_name, const string& config_file_name, const string& detector_file_name);
    ~CameraSystemCalibration();
private:
};

#endif // CAMERA_SYSTEM_CALIBRATION_H

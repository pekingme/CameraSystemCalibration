#include "camera_system_calibration.h"
#include "utils.h"

CameraSystemCalibration::CameraSystemCalibration(const string& output_file_name, const string& config_file_name, const string& detector_file_name)
{
    cout << "Loading camera calibration system configurations ..." << endl;
    
    // check if input files exist
    if(!Utils::fileExists(config_file_name) || !Utils::fileExists(detector_file_name)) exit(-1);
    cout << "\tCalibration configuration file: " << config_file_name << endl;
    cout << "\tArUco detector setting file: " << detector_file_name << endl;
    cout << "\tCalibration result file: " << output_file_name << endl;
    
    
}

CameraSystemCalibration::~CameraSystemCalibration()
{

}

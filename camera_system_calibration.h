#ifndef CAMERA_SYSTEM_CALIBRATION_H
#define CAMERA_SYSTEM_CALIBRATION_H

#include <string>
#include <vector>
#include "structs.h"
#include "video_clip.h"
#include "camera_calibration.h"
#include "opencv2/opencv.hpp"
#include "opencv2/aruco/charuco.hpp"

using namespace std;
using namespace cv;

class CameraSystemCalibration
{
public:
    CameraSystemCalibration ( const string& output_file_name, const string& config_file_name, const string& detector_file_name );
    ~CameraSystemCalibration() {};
    
    void FetchCornersInFrames();
    void Calibrate();
private:
    // Reads all video file names and camera names from calibration setting file.
    void ReadVideoFileAndCameraNames ( const string& config_file_name, vector<string>* camera_names, vector<string>* video_file_names );
    
    // Reads all calibration parameters from calibration setting file.
    void ReadCalibrationParameters(const string& config_file_name, CameraSystemCalibrationOptions* options);
    
    // Reads all Aruco marker detector parameters from detector setting file.
    void ReadArucoParameters(const string& aruco_file_name, CameraSystemCalibrationOptions* options);
    
    // Performs calibration on each camera individually.
    void CalibrateMonoCameras();
    
    // Aligns all cameras and boards based on initial extrinsic parameters.
    void AlignCamerasAndBoards();
    
    // Optimizes poses of all cameras and boards to minimize reprojection error.
    void OptimizeExtrinsics();
    
    // Calculates reprojection error of current calibration result.
    void Reproject();
    
    const string _output_file_name;
    vector<VideoClip> _synchronized_video_clips;
    CameraSystemCalibrationOptions _options;
    vector<CameraCalibration> _camera_calibrations;
};

#endif // CAMERA_SYSTEM_CALIBRATION_H
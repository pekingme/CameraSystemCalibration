#ifndef CAMERA_SYSTEM_CALIBRATION_H
#define CAMERA_SYSTEM_CALIBRATION_H

#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
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
    CameraSystemCalibration ( const string& output_folder, const string& config_file_name, const string& detector_file_name,
                              const string& mono_file_name, const bool mono_calibration_used, const bool ceres_details_enabled );
    ~CameraSystemCalibration() {};

    // Extract charuco corners from sampled frames of videos.
    void FetchCornersInFrames();

    // Calibrates the camera system.
    void Calibrate();

    // Saves calibration results to output folder.
    void SaveResults();

    // Saves calibration results to output folder in MultiColSLAM format.
    void SaveResults2();
private:
    // Reads all video file names and camera names from calibration setting file.
    void ReadVideoFileAndCameraNames ( const string& config_file_name, vector<string>* camera_names, vector<string>* video_file_names );

    // Reads all calibration parameters from calibration setting file.
    void ReadCalibrationParameters ( const string& config_file_name, CameraSystemCalibrationOptions* options );

    // Reads all Aruco marker detector parameters from detector setting file.
    void ReadArucoParameters ( const string& aruco_file_name, CameraSystemCalibrationOptions* options );

    // Reads mono camera calibration from file.
    void ReadMonoCalibrations ( const string& mono_file_name );

    // Performs calibration on each camera individually.
    void CalibrateMonoCameras();

    // Aligns all cameras and boards based on initial extrinsic parameters.
    void AlignCamerasAndBoards();

    // Optimizes poses of all cameras and boards to minimize reprojection error.
    void OptimizeExtrinsics();
    
    // Optimizes poses and models of all cameras and boards to minimize reprojection error.
    void OptimizeFully();

    // Calculates reprojection error of current calibration result.
    double Reproject();

    // Folder storing final calibration results.
    const string _output_folder;

    // Vector of videos used in the calibration.
    vector<VideoClip> _synchronized_video_clips;

    // Calibration settings.
    CameraSystemCalibrationOptions _options;

    // Vector of single camera calibration objects.
    vector<CameraCalibration> _camera_calibrations;

    // Vector of camera's transform to camera system center.
    vector<Mat> _camera_extrinsics;

    // Map of poses of all vertices (cameras and frames).
    unordered_map<string, Mat> _vertex_pose_map;

    // Whether mono camera calibration result is used.
    bool _mono_calibration_used;

    // Toggle ceres progress to console.
    const bool _ceres_details_enabled;

    // Whether camera system has been calibrated.
    bool _calibrated;
    
    // Map to hold all provided camera intrinsic calibration parameters.
    unordered_map<string, Camera> _provided_cameras;
};

#endif // CAMERA_SYSTEM_CALIBRATION_H

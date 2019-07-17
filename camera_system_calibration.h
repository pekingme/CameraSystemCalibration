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
    CameraSystemCalibration() : _mono_calibration_used ( false ), _calibrated ( false ) {}
    ~CameraSystemCalibration() {};

    // Load configuration file and parameters for camera calibration with videos.
    void LoadCalibrationWithVideo ( const string& output_folder, const string& config_file_name, const string& detector_file_name, const bool ceres_details_enabled );

    // Load configuration file and parameters for camera calibration with videos and mono calibration result.
    void LoadCalibrationWithVideoAndMono ( const string& output_folder, const string& config_file_name, const string& detector_file_name,
                                           const string& mono_file_name, const bool ceres_details_enabled );

    // Load configuration file and parameters for camera calibration with images.
    void LoadCalibrationWithPhotos ( const string& output_folder, const string& config_file_name, const string& detector_file_name, const bool ceres_details_enabled );

    // Extract frames from video and charuco corners from them.
    void ExtractFramesAndCorners();

    // Calibrates the camera system.
    void Calibrate();

    // Saves camera intrinsic parameters to output folder.
    void SaveIntrinsics();

    // Saves inter-camera extrinsic parameters to output folder.
    void SaveInterCameraExtrinsics();

    // Saves calibration results to output folder.
    void SaveResultsCombined();

private:
    // Reads all video file names and camera names from calibration setting file.
    void ReadVideoFileAndCameraNames ( const string& config_file_name, vector<string>* camera_names, vector<string>* video_file_names );

    // Reads all photo folders and camera names from calibration setting file.
    void ReadPhotoFoldersAndCameraNames ( const string& config_file_name, vector<string>* camera_names, vector<string>* photo_folders );

    // Reads all calibration parameters from calibration setting file.
    void ReadCalibrationParameters ( const string& config_file_name, CameraSystemCalibrationOptions* options );

    // Reads all Aruco marker detector parameters from detector setting file.
    void ReadArucoParameters ( const string& aruco_file_name, CameraSystemCalibrationOptions* options );

    // Reads mono camera calibration from file.
    void ReadMonoCalibrations ( const string& mono_file_name, unordered_map<string, Camera>* provided_cameras );

    // Performs calibration on each camera individually.
    void CalibrateMonoCameras();

    // Aligns all cameras and boards based on initial extrinsic parameters.
    // Returns if there're enough frames shared for extrinsic optimization.
    bool AlignCamerasAndBoards();

    // Optimizes poses of all cameras and boards to minimize reprojection error.
    void OptimizeExtrinsics();

    // Optimizes poses and models of all cameras and boards to minimize reprojection error.
    void OptimizeFully();

    // Calculates reprojection error of current calibration result.
    double Reproject();

    // Folder storing final calibration results.
    string _output_folder;

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
    bool _ceres_details_enabled;

    // Whether camera system has been calibrated.
    bool _calibrated;

    // Map to hold all provided camera intrinsic calibration parameters.
    unordered_map<string, Camera> _provided_cameras;
};

#endif // CAMERA_SYSTEM_CALIBRATION_H

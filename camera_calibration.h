#ifndef CAMERACALIBRATION_H
#define CAMERACALIBRATION_H

#include <string>
#include <vector>
#include <unordered_set>
#include "utils.h"
#include "structs.h"
#include "optimization_structs.h"
#include "camera.h"
#include "video_clip.h"
#include "opencv2/aruco/charuco.hpp"
#include "ceres/ceres.h"

using namespace cv;
using namespace std;

class CameraCalibration
{
public:
    CameraCalibration ( const string& camera_name, const CameraSystemCalibrationOptions& options, bool ceres_details_enabled )
        : _options ( options ), _camera ( camera_name ), _ceres_details_enabled(ceres_details_enabled) {};

    // Extracts corners from frame and save if it's valid.
    bool ExtractCornersAndSave(Frame* frame);
    
    // Performs initial calibration.
    void Calibrate();
    
    // Rejects frames with very big error.
    void RejectFrames(const double average_bound, const double deviation_bound);
    
    // Optimizes intrinsic and extrinsic parameters.
    void OptimizeFully();
    
    // Calculates reprojection error of current calibration result.
    void Reproject();
	
    Camera* GetCameraPtr() {
        return &_camera;
    }
private:
    // Calculates simplified transform matrix [r11, r12, t1; r21, r22, t2; r31, r32, 0] for each frame.
    // Since z of board corners is all 0, vector r3 can be ignored for now.
    // t3 is set to 0 and will be determined later.
    void CalculateInitialTransforms(const Frame& frame, Mat* u, Mat* v, Mat* x, Mat* y, vector<Mat>* transforms);
    
    // Filters out invalid transforms based on translation for each frame.
    bool RefineTransforms(const Mat& u, const Mat& v, const Mat& x, const Mat& y, vector<Mat>* transforms);
    
    // Finalizes the transform based on the fact that the center must be minima for each frame.
    bool FinalizeTransform(const Mat& u, const Mat& v, const Mat& x, const Mat& y, const vector<Mat>& transforms, Mat* final_transform);
  
    // Calculates poly parameters and t3 together using Ceres for all frames.
    void CalculatePolyAndT3();
    
    // Calculates inverse poly parameters from poly parameters.
    void CalculateInversePoly();
    
    const CameraSystemCalibrationOptions _options;
    Camera _camera;
    const bool _ceres_details_enabled;
    unordered_set<unsigned> _valid_frame_set;
    vector<Frame> _valid_frames;
};

#endif // CAMERACALIBRATION_H

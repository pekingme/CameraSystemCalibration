#ifndef CAMERACALIBRATION_H
#define CAMERACALIBRATION_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include "utils.h"
#include "structs.h"
#include "optimization_structs.h"
#include "camera.h"
#include "video_clip.h"
#include "opencv2/aruco/charuco.hpp"

using namespace cv;
using namespace std;

class CameraCalibration
{
public:
    CameraCalibration ( const string& camera_name, const CameraSystemCalibrationOptions& options, bool ceres_details_enabled )
        : _options ( options ), _camera ( camera_name ), _ceres_details_enabled ( ceres_details_enabled ) {}
    CameraCalibration ( const string& camera_name, const CameraSystemCalibrationOptions& options, const Camera& camera, bool ceres_details_enabled )
        : _options ( options ), _camera ( camera ), _ceres_details_enabled ( ceres_details_enabled ) {}

    // Extracts corners from frame and save if it's valid.
    bool ExtractCornersAndSave ( Frame* frame );

    // Overrides the intrinsic parameters with input values.
    // Calculates transform for each frame with loaded intrinsic parameters.
    void OverrideIntrinsicsAndUpdateExtrinsics ( const Camera& new_camera );

    // Performs mono calibration.
    void Calibrate();

    // Rejects frames with very big error.
    int RejectFrames ( const double average_bound, const double deviation_bound, const bool remove_invalid, const bool show_error );

    // Revalidate all frames.
    void RevalidateFrames();

    // Optimizes intrinsic and extrinsic parameters.
    void OptimizeFully ();

    // Optimizes extrinsic parameters.
    void OptimizeExtrinsic ();

    // Calculates and return reprojection error per corner of current calibration result.
    double Reproject();

    // Saves all valid frames with/without detected corners and/or reprojected corners.
    void SaveAllValidFrames ( const string& folder, const bool draw_detected, const bool draw_reprojected, const bool save_original );

    // Returns the name of current camera model.
    string GetCameraName() {
        return GetCameraPtr()->GetName();
    }

    // Returns the pointer of current camera model's object.
    Camera* GetCameraPtr() {
        return &_camera;
    }

    // Returns the pointer of valid frames vector.
    vector<Frame>* GetValidFramesPtr() {
        return &_valid_frames;
    }

    // Returns the pointer of the valid frame with indicated global index.
    Frame* GetFrameWithGloablIndex ( unsigned global_index ) {
        return &_valid_frames[_global_to_local_map[global_index]];
    }
private:
    // Calculates simplified transform matrix [r11, r12, t1; r21, r22, t2; r31, r32, 0] for each frame.
    // Since z of board corners is all 0, vector r3 can be ignored for now.
    // t3 is set to 0 and will be determined later.
    void CalculateInitialTransforms ( const Frame& frame, Mat* u, Mat* v, Mat* x, Mat* y, vector<Mat>* transforms );

    // Filters out invalid transforms based on translation for each frame.
    bool RefineTransforms ( const Mat& u, const Mat& v, const Mat& x, const Mat& y, vector<Mat>* transforms );

    // Finalizes the transform based on the fact that the center must be minima for each frame.
    bool FinalizeTransform ( const Mat& u, const Mat& v, const Mat& x, const Mat& y, const vector<Mat>& transforms, Mat* final_transform );

    // Calculates poly parameters and t3 together using Ceres for all frames.
    void CalculatePolyAndT3();

    // Calculates inverse poly parameters from poly parameters.
    void CalculateInversePolyFromPoly();

    // Calculates poly parameters from inverse poly parameters.
    void CalculatePolyFromInversePoly();

    // Calculates extrinsic parameters with provided intrinsics.
    // See eq. 10.1, 10.2, 10.3 in Scaramuzza's paper.
    void CalculateExtrinsicWithIntrinsic ( Frame* frame );

    // Removes all frame objects which are no longer valid.
    void RemoveInvalidFrames();

    // Calibration settings.
    const CameraSystemCalibrationOptions _options;

    // Camera model.
    Camera _camera;

    // Toggle ceres progress to console.
    const bool _ceres_details_enabled;

    // Map from frame global index to local index in _valid_frames.
    unordered_map<unsigned, unsigned> _global_to_local_map;

    // Vector of valid frames.
    vector<Frame> _valid_frames;
};

#endif // CAMERACALIBRATION_H

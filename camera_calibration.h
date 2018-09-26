#ifndef CAMERACALIBRATION_H
#define CAMERACALIBRATION_H

#include <string>
#include <vector>
#include <unordered_set>
#include "utils.h"
#include "structs.h"
#include "camera.h"
#include "video_clip.h"
#include "opencv2/aruco/charuco.hpp"

using namespace cv;
using namespace std;

class CameraCalibration
{
public:
    CameraCalibration ( const string& camera_name, const CameraSystemCalibrationOptions& options )
        : _options ( options ), _camera ( camera_name ) {};

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
    const CameraSystemCalibrationOptions _options;
    Camera _camera;
    unordered_set<unsigned> _valid_frame_set;
    vector<Frame> _valid_frames;
};

#endif // CAMERACALIBRATION_H

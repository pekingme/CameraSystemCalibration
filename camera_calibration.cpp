#include "camera_calibration.h"

bool CameraCalibration::ExtractCornersAndSave ( Frame* frame )
{
    // Extracts aruco markers.
    vector<int> marker_ids;
    vector<vector<Point2f>> marker_detected_corners, marker_rejected_corners;
    aruco::detectMarkers ( frame->original_frame, _options.dictionary, marker_detected_corners, marker_ids, _options.detector_parameters, marker_rejected_corners );
    if ( marker_ids.size() == 0 )
    {
        return false;
    }

    // Interpolate detected charuco corners.
    const Ptr<aruco::CharucoBoard> board = _options.charuco_board;
    Mat charuco_detected_corners, charuco_corners_ids;
    aruco::interpolateCornersCharuco ( marker_detected_corners, marker_ids, frame->original_frame, board, charuco_detected_corners, charuco_corners_ids );
    Mat flatten_charuco_detected_corners = charuco_detected_corners.reshape ( 1 ); // Flatten N x 1 x 32FC2 to N x 2 x 32FC1.
    if ( charuco_corners_ids.total() < 12 )
    {
        return false;
    }

    // Collects board charuco corners.
    Mat charuco_board_corners = Utils::GetCharucoBoardCornersMatFromVector ( charuco_corners_ids, board->chessboardCorners );
    Mat flatten_charuco_board_corners = charuco_board_corners.reshape ( 1 ); // Flatten N x 1 x 32FC3 to N x 3 x 32FC1.

    // Updates corner information in Frame.
    frame->corner_count = charuco_corners_ids.total();
    frame->corner_ids = charuco_corners_ids;
    frame->board_corners_32 = charuco_board_corners;
    frame->detected_corners_32 = charuco_detected_corners;
    flatten_charuco_board_corners.convertTo ( frame->flatten_board_corners_64, CV_64F );
    flatten_charuco_detected_corners.convertTo ( frame->flatten_detected_corners_64, CV_64F );
    // TODO add transform34 to frame
    frame->valid = true;
    
    return true;
}

void CameraCalibration::Calibrate()
{

}

void CameraCalibration::RejectFrames ( const double average_bound, const double deviation_bound )
{

}

void CameraCalibration::OptimizeFully()
{

}

void CameraCalibration::Reproject()
{

}

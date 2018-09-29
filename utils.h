#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <iostream>
#include "structs.h"
#include "camera.h"
#include "opencv2/opencv.hpp"
#include "opencv2/aruco/charuco.hpp"

using namespace std;
using namespace cv;

class Utils
{
public:
    // Returns if a file exists in file system.
    static bool FileExists ( const string& filename );
    
    // Draws a frame on a named window with/without detected corners and/or reprojected corners.
    static void ShowFrameInWindow( const string& window_name, const Frame& frame, const bool draw_corners, const bool draw_reprojections);
    
    // Collects points from vector based on ids to a N x 1 x c(3) Mat.
    static Mat GetCharucoBoardCornersMatFromVector(const Mat& corner_ids, const vector<Point3f>& point_vector);
    
    // Return +1 if positive, -1 if negative, 0 otherwise.
    static int Sign(const double value);
    
    // Evaluates polynomial equations.
    static double EvaluatePolyEquation(const double* coefficients, const int n, const double x);
    
    // Fetches rodrigues rotation vector and translation vector from 3 x 4 transform matrix.
    static void GetRAndTVectorsFromTransform(const Mat& transform, Mat* r_mat, Mat* t_mat);
    
    // Forms up 3 x 4 transform matrix from rotation vector and translation vector.
    static void GetTransformFromRAndTVectors(const Mat& r_mat, const Mat& t_mat, Mat* transform);
    
    // Reprojects the board corners in the frame based on input intrinsic and extrinsic parameters.
    static void ReprojectCornersInFrame(const double* intrinsics, const double* rotation_vector, const double* translation_vector, 
					const Mat& flatten_board_corners, Mat* reprojected_corners);
};

#endif // UTILS_H

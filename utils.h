#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <iostream>
#include "structs.h"
#include "opencv2/opencv.hpp"

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
};

#endif // UTILS_H

#include "utils.h"

bool Utils::FileExists ( const string& filename )
{
    if ( FILE* file = fopen ( filename.c_str(), "r" ) )
    {
        fclose ( file );
        return true;
    }
    else
    {
        cerr << "File not found: " << filename << endl;
        return false;
    }
}

void Utils::ShowFrameInWindow ( const string& window_name, const Frame& frame, const bool draw_corners, const bool draw_reprojections )
{
    Mat canvas_mat;
    frame.original_frame.copyTo ( canvas_mat );
    // TODO draw corners
    imshow ( window_name, canvas_mat );
}

Mat Utils::GetCharucoBoardCornersMatFromVector ( const Mat& corner_ids, const vector< Point3f >& point_vector )
{
    Mat charuco_board_corners ( corner_ids.total(), 1, CV_32FC3 );
    for ( size_t p=0; p<corner_ids.total(); p++ )
    {
        charuco_board_corners.at<Vec3f> ( p,0 ) [0] = point_vector[corner_ids.at<int> ( p,0 )].x;
        charuco_board_corners.at<Vec3f> ( p,0 ) [1] = point_vector[corner_ids.at<int> ( p,0 )].y;
        charuco_board_corners.at<Vec3f> ( p,0 ) [2] = point_vector[corner_ids.at<int> ( p,0 )].z;
    }
    return charuco_board_corners;
}

int Utils::Sign ( const double value )
{
    if ( value > 0 )
    {
        return 1;
    }
    else if ( value < 0 )
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

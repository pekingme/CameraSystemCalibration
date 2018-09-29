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
    if ( draw_corners )
    {
        aruco::drawDetectedCornersCharuco ( canvas_mat, frame.detected_corners_32, frame.corner_ids, Scalar ( 0, 0, 255 ) );
    }
    if ( draw_reprojections )
    {
        //TODO draw reprojections
    }
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

double Utils::EvaluatePolyEquation ( const double* coefficients, const int n, const double x )
{
    double y = 0.0;
    double x_i = 1.0;
    for ( int power=0; power<n; power++ )
    {
        y += coefficients[power] * x_i;
        x_i *= x;
    }
    return y;
}

void Utils::GetRAndTVectorsFromTransform ( const Mat& transform, Mat* r_mat, Mat* t_mat )
{
    CV_Assert ( transform.rows == 3 && transform.cols == 4 );
    Rodrigues ( transform.colRange ( 0, 3 ), *r_mat );
    transform.col ( 3 ).copyTo ( *t_mat );
}

void Utils::GetTransformFromRAndTVectors ( const Mat& r_mat, const Mat& t_mat, Mat* transform )
{
    CV_Assert ( transform->rows == 3 && transform->cols == 4 );
    Rodrigues ( r_mat, transform->colRange ( 0, 3 ) );
    t_mat.copyTo ( transform->col ( 3 ) );
}

void Utils::ReprojectCornersInFrame ( const double* intrinsics, const double* rotation_vector_data,
                                      const double* translation_vector_data, const Mat& flatten_board_corners,
                                      Mat* reprojected_corners )
{
    CV_Assert ( flatten_board_corners.cols == 3 );
    CV_Assert ( flatten_board_corners.type() == CV_64FC1 );

    reprojected_corners->create ( flatten_board_corners.rows, 2, CV_64F );
    Vec3d rotation_vector ( rotation_vector_data[0], rotation_vector_data[1], rotation_vector_data[2] );
    Vec3d translation_vector ( translation_vector_data[0], translation_vector_data[1], translation_vector_data[2] );
    Mat rotation_matrix;
    Rodrigues ( rotation_vector, rotation_matrix );
    Mat flatten_board_corners_in_camera = flatten_board_corners * rotation_matrix.t()
                                          + Mat::ones ( flatten_board_corners.rows, 1, CV_64F ) * Mat ( translation_vector ).t();
    double affine_c = intrinsics[0];
    double affine_d = intrinsics[1];
    double affine_e = intrinsics[2];
    double u0 = intrinsics[3];
    double v0 = intrinsics[4];
    double inverse_poly[INV_POLY_SIZE];
    copy ( intrinsics+INV_POLY_START, intrinsics+INV_POLY_START+INV_POLY_SIZE, inverse_poly );

    for ( int i=0; i<flatten_board_corners.rows; i++ )
    {
        // x, y, z of corner in camera coordinate system.
        double corner_x = flatten_board_corners_in_camera.at<double> ( i, 0 );
        double corner_y = flatten_board_corners_in_camera.at<double> ( i, 1 );
        double corner_z = flatten_board_corners_in_camera.at<double> ( i, 2 );
        // Calculates norm on xy plane.
        double norm_on_xy_2 = corner_x * corner_x + corner_y * corner_y;
        double norm_on_xy = 0.0;
        if ( norm_on_xy_2 > 0.0 )
        {
            norm_on_xy = sqrt ( norm_on_xy_2 );
        }
        // Calculates projected position on the frame.
        double theta = atan2 ( corner_z, norm_on_xy );
        double rho = EvaluatePolyEquation ( inverse_poly, INV_POLY_SIZE, theta );
        // u, v of corner on frame without affine.
        double corner_u = corner_x / norm_on_xy * rho;
        double corner_v = corner_y / norm_on_xy * rho;
        // Affines corner on frame.
        reprojected_corners->at<double> ( i, 0 ) = affine_c * corner_u + affine_d * corner_v + u0;
        reprojected_corners->at<double> ( i, 1 ) = affine_e * corner_u + corner_v + v0;
    }
}

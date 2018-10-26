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

void Utils::ShowFrameInWindow ( const string& window_name, const Frame& frame, const bool draw_detected, const bool draw_reprojected )
{
    Mat canvas_mat;
    frame.original_frame.copyTo ( canvas_mat );
    if ( draw_detected && !frame.detected_corners_32.empty() )
    {
        aruco::drawDetectedCornersCharuco ( canvas_mat, frame.detected_corners_32, frame.corner_ids, Scalar ( 0, 0, 255 ) );
    }
    if ( draw_reprojected && !frame.reprojected_corners_32.empty() )
    {
        aruco::drawDetectedCornersCharuco ( canvas_mat, frame.reprojected_corners_32, frame.corner_ids, Scalar ( 255, 0, 0 ) );
    }
    imshow ( window_name, canvas_mat );
}

void Utils::SaveFrame ( const string& folder, const Frame& frame, const bool draw_detected, const bool draw_reprojected, const bool draw_info )
{
    Mat canvas_mat;
    frame.original_frame.copyTo ( canvas_mat );
    if ( draw_detected && !frame.detected_corners_32.empty() )
    {
        aruco::drawDetectedCornersCharuco ( canvas_mat, frame.detected_corners_32, frame.corner_ids, Scalar ( 0, 0, 255 ) );
    }
    if ( draw_reprojected && !frame.reprojected_corners_32.empty() )
    {
        aruco::drawDetectedCornersCharuco ( canvas_mat, frame.reprojected_corners_32, frame.corner_ids, Scalar ( 255, 0, 0 ) );
    }
    if ( draw_info )
    {
        rectangle ( canvas_mat, Point ( 0, 0 ), Point ( canvas_mat.cols, 40 ), Scalar ( 255, 255, 255 ), CV_FILLED );
        putText ( canvas_mat, "Average reprojection error = "+to_string ( frame.reprojection_error ), Point ( 5, 35 ),  CV_FONT_HERSHEY_SIMPLEX, 1, Scalar ( 255, 0, 0 ), 2 );
    }
    imwrite ( folder + frame.camera_name + "-" + to_string ( frame.global_index ) + ".jpg", canvas_mat );
}

Mat Utils::GetCharucoBoardCornersMatFromVector ( const Mat& corner_ids, const vector< Point3f >& point_vector )
{
    Mat charuco_board_corners ( corner_ids.total(), 1, CV_32FC3 );
    for ( size_t p=0; p<corner_ids.total(); p++ )
    {
        int corner_id = corner_ids.at<int> ( p, 0 );
        Point3f board_corner = point_vector[corner_id];
        charuco_board_corners.at<Vec3f> ( p,0 ) [0] = board_corner.x;
        charuco_board_corners.at<Vec3f> ( p,0 ) [1] = -board_corner.y;
        charuco_board_corners.at<Vec3f> ( p,0 ) [2] = 0;
    }
    return charuco_board_corners;
}

Mat Utils::SwapPointsXandY ( const Mat& points )
{
    Mat out = Mat::zeros ( points.rows, points.cols, points.type() );
    for ( int i=0; i<points.cols; i++ )
    {
        if ( i==0 )
        {
            out.col ( i ) += points.col ( 1 );
        }
        else if ( i==1 )
        {
            out.col ( i ) += points.col ( 0 );
        }
        else
        {
            out.col ( i ) += points.col ( i );
        }
    }
    return out;
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
    for ( int power=n-1; power>=0; --power )
    {
        y = y * x + coefficients[power];
    }
    return y;
}

void Utils::GetRAndTVectorsFromTransform ( const Mat& transform, Mat* r_mat, Mat* t_mat )
{
    CV_Assert ( transform.rows == 3 && transform.cols == 4 );
    r_mat->create ( 3, 1, CV_64FC1 );
    t_mat->create ( 3, 1, CV_64FC1 );
    Rodrigues ( transform.colRange ( 0, 3 ), *r_mat );
    transform.col ( 3 ).copyTo ( *t_mat );
}

void Utils::GetTransformFromRAndTVectors ( const Mat& r_mat, const Mat& t_mat, Mat* transform )
{
    CV_Assert ( transform->rows == 3 && transform->cols == 4 );
    Rodrigues ( r_mat, transform->colRange ( 0, 3 ) );
    t_mat.copyTo ( transform->col ( 3 ) );
}

void Utils::GetCAndTVectorsFromTransform ( const Mat& transform, Mat* c_mat, Mat* t_mat )
{
    CV_Assert ( transform.rows == 3 && transform.cols == 4 );
    c_mat->create ( 3, 1, CV_64FC1 );
    t_mat->create ( 3, 1, CV_64FC1 );
    Mat R = transform.colRange ( 0, 3 );
    GetCVectorFromRotation ( R, c_mat );
    transform.col ( 3 ).copyTo ( *t_mat );
}

void Utils::GetCVectorFromRotation ( const Mat& rotation, Mat* c_mat )
{
    CV_Assert ( rotation.rows == 3 && rotation.cols == 3 );
    Mat eye = Mat::eye ( 3, 3, rotation.type() );
    Mat c1 = rotation - eye;
    Mat c2 = rotation + eye;
    Mat c = c1 * c2.inv();

    c_mat->at<double> ( 0, 0 ) = -c.at<double> ( 1, 2 );
    c_mat->at<double> ( 1, 0 ) = c.at<double> ( 0, 2 );
    c_mat->at<double> ( 2, 0 ) = -c.at<double> ( 0, 1 );
}

Mat Utils::GetTransform44From34 ( const Mat& transform_3_4 )
{
    CV_Assert ( transform_3_4.rows == 3 && transform_3_4.cols == 4 );
    Mat transform_4_4 = Mat::eye ( 4, 4, transform_3_4.type() );
    transform_3_4.copyTo ( transform_4_4.rowRange ( 0, 3 ) );
    return transform_4_4;
}

Mat Utils::InvertTransform ( const Mat& transform_3_4 )
{
    CV_Assert ( transform_3_4.rows == 3 && transform_3_4.cols == 4 );
    Mat rotation = transform_3_4.colRange ( 0, 3 );
    Mat inverted_rotation = rotation.t();
    Mat translation = transform_3_4.col ( 3 );
    Mat inverted_translation = -inverted_rotation * translation;
    Mat inverted_transform = Mat::eye ( 3, 4, transform_3_4.type() );
    inverted_rotation.copyTo ( inverted_transform.colRange ( 0, 3 ) );
    inverted_translation.copyTo ( inverted_transform.col ( 3 ) );

    return inverted_transform;
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
        double norm_on_xy = hypot ( corner_x, corner_y );
        if ( norm_on_xy == 0.0 )
        {
            norm_on_xy = 1e-14;
        }
        // Calculates projected position on the frame. Flip z.
        double theta = atan2 ( -corner_z, norm_on_xy );
        double rho = EvaluatePolyEquation ( inverse_poly, INV_POLY_SIZE, theta );
        // u, v of corner on frame without affine.
        double corner_u = corner_x * rho / norm_on_xy;
        double corner_v = corner_y * rho / norm_on_xy;

        // Affines corner on frame.
        reprojected_corners->at<double> ( i, 0 ) = affine_c * corner_u + affine_d * corner_v + u0;
        reprojected_corners->at<double> ( i, 1 ) = affine_e * corner_u + corner_v + v0;
    }
}

void Utils::ReprojectSingleCorner ( const double* intrinsics, const double* rotation_vector_data,
                                    const double* translation_vector_data, const Vec3d& board_corner, Vec2d* reprojected_corner )
{
    Vec3d rotation_vector ( rotation_vector_data[0], rotation_vector_data[1], rotation_vector_data[2] );
    Vec3d translation_vector ( translation_vector_data[0], translation_vector_data[1], translation_vector_data[2] );
    Mat rotation_matrix;
    Rodrigues ( rotation_vector, rotation_matrix );
    Vec3d board_corner_in_camera ( ( double* ) Mat ( rotation_matrix * Mat ( board_corner ) + Mat ( translation_vector ) ).data );

    double affine_c = intrinsics[0];
    double affine_d = intrinsics[1];
    double affine_e = intrinsics[2];
    double u0 = intrinsics[3];
    double v0 = intrinsics[4];
    double inverse_poly[INV_POLY_SIZE];
    copy ( intrinsics+INV_POLY_START, intrinsics+INV_POLY_START+INV_POLY_SIZE, inverse_poly );

    // Calculates norm on xy plane.
    double norm_on_xy = hypot ( board_corner_in_camera[0], board_corner_in_camera[1] );
    if ( norm_on_xy == 0.0 )
    {
        norm_on_xy = 1e-14;
    }
    // Calculates projected position on the frame. Flip z.
    double theta = atan2 ( -board_corner_in_camera[2], norm_on_xy );
    double rho = EvaluatePolyEquation ( inverse_poly, INV_POLY_SIZE, theta );
    // u, v of corner on frame without affine.
    double corner_u = board_corner_in_camera[0] * rho / norm_on_xy;
    double corner_v = board_corner_in_camera[1] * rho / norm_on_xy;

    // Affines corner on frame.
    ( *reprojected_corner ) [0] = affine_c * corner_u + affine_d * corner_v + u0;
    ( *reprojected_corner ) [1] = affine_e * corner_u + corner_v + v0;
}

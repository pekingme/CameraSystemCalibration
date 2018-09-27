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
    frame->transform = Mat::zeros ( 3, 4, CV_64FC1 );
    frame->valid = true;

    return true;
}

void CameraCalibration::Calibrate()
{
    if ( _valid_frame_set.size() == 0 )
    {
        cerr << "No valid frames are found for mono calibration." << endl;
        exit ( -1 );
    }
    // Initializes intrinsic parameters.
    double initial_intrinsics[TOTAL_SIZE] = {1, 0, 0, _camera.GetWidth() /2.0, _camera.GetHeight() /2.0};
    _camera.SetIntrinsics ( initial_intrinsics );
    // Initializes extrinsic parameters based on observed corners.
    for ( unsigned i=0; i<_valid_frames.size(); i++ )
    {
        vector<Mat> transforms;
        Mat u, v, x, y;
        CalculateInitialTransforms ( _valid_frames[i], &u, &v, &x, &y, &transforms );
        if ( RefineTransforms ( u, v, x, y, &transforms )
                || FinalizeTransform ( u, v, x, y, transforms, &_valid_frames[i].transform ) )
        {
            _valid_frames[i].valid = false;
            continue;
        }
    }
}

void CameraCalibration::CalculateInitialTransforms ( const Frame& frame, Mat* u, Mat* v, Mat* x, Mat* y, vector< Mat >* transforms )
{
    // Splits corners coordinates into u, v, x, y.
    Mat ( frame.flatten_detected_corners_64.col ( 0 ) - _camera.GetU0() ).copyTo ( *u );
    Mat ( frame.flatten_detected_corners_64.col ( 1 ) - _camera.GetV0() ).copyTo ( *v );
    Mat ( frame.flatten_board_corners_64.col ( 0 ) ).copyTo ( *x );
    Mat ( frame.flatten_board_corners_64.col ( 1 ) ).copyTo ( *y );

    // Forms up a matrix to solve r11, r12, r21, r22, t1, t2. (Matrix M in Scaramuzza's paper)
    Mat m ( frame.corner_count, 6, CV_64F, 0.0 );
    m.col ( 0 ) += -v->mul ( *x );
    m.col ( 1 ) += -v->mul ( *y );
    m.col ( 2 ) += u->mul ( *x );
    m.col ( 3 ) += u->mul ( *y );
    m.col ( 4 ) += -*v;
    m.col ( 5 ) += *u;

    // Solves m for r11, r12, r21, r22, t1, t2.
    SVD svd ( m, CV_SVD_MODIFY_A );
    double r11 = svd.vt.at<double> ( 5,0 );
    double r12 = svd.vt.at<double> ( 5,1 );
    double r21 = svd.vt.at<double> ( 5,2 );
    double r22 = svd.vt.at<double> ( 5,3 );
    double t1 = svd.vt.at<double> ( 5,4 );
    double t2 = svd.vt.at<double> ( 5,5 );

    // Substitutes above to r1 x r2 = 0 and |r1| = |r2| = 1, we get
    // r32^4 + (r12^2 + r22^2 - r11^2 - r21^2) r32^2 - (r11*r12 + r21*r22)^2 = 0
    // Solves the quadratic equation above to get r32^2 and r31^2.
    double a = 1;
    double b = r12*r12 + r22*r22 - r11*r11 - r21*r21;
    double c = - ( r11*r12 + r21*r22 ) * ( r11*r12 + r21*r22 );
    double d = sqrt ( b*b - 4*a*c );
    double r32_2 = ( -b + d ) / ( 2*a );
    double r32_temp = sqrt ( r32_2 );

    // Here are all possible r31 and r32.
    vector<double> r31s, r32s;
    if ( r32_temp * r32_temp == 0 )
    {
        r32s.push_back ( r32_temp );
        r31s.push_back ( sqrt ( b ) );
        r32s.push_back ( r32_temp );
        r31s.push_back ( -sqrt ( b ) );
        r32s.push_back ( -r32_temp );
        r31s.push_back ( sqrt ( b ) );
        r32s.push_back ( -r32_temp );
        r31s.push_back ( -sqrt ( b ) );
    }
    else
    {
        r32s.push_back ( r32_temp );
        r31s.push_back ( - ( r11*r12 + r21*r22 ) / r32_temp );
        r32s.push_back ( -r32_temp );
        r31s.push_back ( ( r11*r12 + r21*r22 ) / r32_temp );
    }

    // Forms up transform matrices.
    for ( unsigned i=0; i<r31s.size(); i++ )
    {
        Vec3d r1 ( r11, r21, r31s[i] );
        Vec3d r2 ( r12, r22, r32s[i] );
        Vec3d t ( t1, t2, 0 );

        double lambda = 1 / norm ( r1 );
        Mat transform1 = Mat::zeros ( 3,3,CV_64F );
        Mat ( r1 ).copyTo ( transform1.col ( 0 ) );
        Mat ( r2 ).copyTo ( transform1.col ( 1 ) );
        Mat ( t ).copyTo ( transform1.col ( 2 ) );
        transform1 = transform1 * lambda;
        transforms->push_back ( transform1 );

        Mat transform2;
        transform1.copyTo ( transform2 );
        transform2 = transform2 * -1;
        transforms->push_back ( transform2 );
    }
}

bool CameraCalibration::RefineTransforms ( const Mat& u, const Mat& v, const Mat& x, const Mat& y, vector< Mat >* transforms )
{
    vector<Mat> refined_transforms;

    // Finds transform with the minimum translation.
    double min_translation = numeric_limits< double >::infinity();
    int min_translation_index = -1;
    for ( unsigned i=0; i<transforms->size(); i++ )
    {
        Vec2d t ( (*transforms)[i].at<double> ( 0,2 ), (*transforms)[i].at<double> ( 1,2 ) );
        Vec2d p ( u.at<double> ( 0,0 ), v.at<double> ( 0,0 ) );
        double translation = norm ( t - p );
        if ( translation < min_translation )
        {
            min_translation = translation;
            min_translation_index = i;
        }
    }

    // Keeps transforms with t vector, which has same sign as minimum translation transform.
    if ( min_translation_index != -1 )
    {
        for ( unsigned i=0; i<transforms->size(); i++ )
        {
            double t1 = (*transforms)[i].at<double> ( 0,2 );
            double t2 = (*transforms)[i].at<double> ( 1,2 );
            double min_t1 = (*transforms)[min_translation_index].at<double> ( 0,2 );
            double min_t2 = (*transforms)[min_translation_index].at<double> ( 1,2 );
            if ( Utils::Sign ( t1 ) == Utils::Sign ( min_t1 ) && Utils::Sign ( t2 ) == Utils::Sign ( min_t2 ) )
            {
                refined_transforms.push_back ( (*transforms)[i] );
            }
        }
        transforms->swap ( refined_transforms );
        return true;
    }
    else
    {
        return false;
    }
}

bool CameraCalibration::FinalizeTransform ( const Mat& u, const Mat& v, const Mat& x, const Mat& y, const vector< Mat >& transforms, Mat* final_transform )
{
    // Selects transform candidates by calculating intrinsic with pseudo-inv.
    // Coefficient of rho^2 must be positive to guarantee the center is minima.
    int final_index = -1;
    for ( unsigned i=0; i<transforms.size(); i++ )
    {
        Mat transform = transforms[i];
        double r11 = transform.at<double> ( 0,0 );
        double r21 = transform.at<double> ( 1,0 );
        double r31 = transform.at<double> ( 2,0 );
        double r12 = transform.at<double> ( 0,1 );
        double r22 = transform.at<double> ( 0,1 );
        double r32 = transform.at<double> ( 0,1 );
        double t1 = transform.at<double> ( 0,2 );
        double t2 = transform.at<double> ( 1,2 );

        // Forms up matrix to solve poly coefficient of rho^2. (see eq.13 in Scaramuzza's paper)
        Mat m_a, m_b, m_c, m_d, rho, rho_2;
        m_a = r21*x + r22*y + t2;
        m_b = v.mul ( r31*x + r32*y );
        m_c = r11*x + r12*y + t1;
        m_d = u.mul ( r31*x + r32*y );
        rho_2 = x.mul ( x ) + y.mul ( y );
        sqrt ( rho_2, rho );

        // Solves the matrix.
        Mat m_a_rho, m_c_rho, pp, qq;
        vector<Mat> m_a_rho_vec = {m_a, m_a.mul ( rho ), m_a.mul ( rho_2 ), -v};
        vector<Mat> m_c_rho_vec = {m_c, m_c.mul ( rho ), m_c.mul ( rho_2 ), -u};
        hconcat ( m_a_rho_vec, m_a_rho );
        hconcat ( m_c_rho_vec, m_c_rho );
        vconcat ( m_a_rho, m_c_rho, pp );
        vconcat ( m_b, m_d, qq );
        Mat pseudo_inv_pp;
        invert ( pp, pseudo_inv_pp, DECOMP_SVD );
        Mat solution = pseudo_inv_pp * qq;

        if ( solution.at<double> ( 2,0 ) > 0 )
        {
            final_index = i;
        }
    }

    if ( final_index == -1 )
    {
        return false;
    }

    Mat r1 = transforms[final_index].col ( 0 );
    Mat r2 = transforms[final_index].col ( 1 );
    Mat r3 = r1.cross ( r2 );
    Mat t = transforms[final_index].col ( 2 );

    r1.copyTo ( final_transform->col ( 0 ) );
    r2.copyTo ( final_transform->col ( 1 ) );
    r3.copyTo ( final_transform->col ( 2 ) );
    t.copyTo ( final_transform->col ( 3 ) );

    return true;
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

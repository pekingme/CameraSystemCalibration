#include "camera_system_calibration.h"
#include "video_synchronizer.h"
#include "utils.h"

CameraSystemCalibration::CameraSystemCalibration ( const string& output_file_name, const string& config_file_name, const string& detector_file_name, const bool ceres_details_enabled )
    : _output_file_name ( output_file_name ), _ceres_details_enabled ( ceres_details_enabled )
{
    cout << "Preparing camera calibration system ..." << endl;

    // Checks if input files exist.
    if ( !Utils::FileExists ( config_file_name ) || !Utils::FileExists ( detector_file_name ) )
    {
        exit ( -1 );
    }
    cout << "\tCalibration configuration file: " << config_file_name << endl;
    cout << "\tArUco detector setting file: " << detector_file_name << endl;
    cout << "\tCalibration result file: " << output_file_name << endl;

    // Gets all calibration and aruco detector parameters.
    cout << endl << "\tLoading calibration and aruco detector settings." << endl;
    ReadCalibrationParameters ( config_file_name, &_options );
    ReadArucoParameters ( detector_file_name, &_options );

    // Gets all video files with camera names and checks all files exist.
    cout << endl << "\tLoading all camera and video names." << endl;
    vector<string> camera_names, video_file_names;
    ReadVideoFileAndCameraNames ( config_file_name, &camera_names, &video_file_names );
    cout << "\tInput video files:" << endl;
    for ( unsigned i=0; i<video_file_names.size(); i++ )
    {
        cout << "\t - [" << camera_names[i] << "] " << video_file_names[i] << endl;
    }

    // Synchronizes all videos.
    cout << endl << "\tSynchronizing videos ..." << endl;
    FileStorage file_storage ( config_file_name, FileStorage::READ );
    double shift_window = file_storage["MaxShift"];
    VideoSynchronizer video_synchronizer;
    video_synchronizer.LoadVideosWithFileNames ( video_file_names, camera_names );
    video_synchronizer.SynchronizeVideoWithAudio ( shift_window*2 );
    _synchronized_video_clips = video_synchronizer.GetVideoClips();
    cout << "\tAll videos are synchronized." << endl;

    // Creates single camera calibration.
    for ( unsigned i=0; i<_synchronized_video_clips.size(); i++ )
    {
        _camera_calibrations.push_back ( CameraCalibration ( _synchronized_video_clips[i].GetCameraName(), _options, ceres_details_enabled ) );
        _camera_extrinsics.push_back ( Mat::eye ( 4, 4, CV_64FC1 ) );
    }
    cout << "... Done." << endl;
}

void CameraSystemCalibration::FetchCornersInFrames()
{
    cout << endl << "Fetching frames and charuco corners from all videos ..." << endl;

    // Loads video files if not, and create windows.
    for ( unsigned i=0; i<_synchronized_video_clips.size(); i++ )
    {
        VideoClip* video_clip = &_synchronized_video_clips[i];
        video_clip->LoadVideoCapture();
        CameraCalibration* camera_calibration = &_camera_calibrations[i];
        Camera* camera = camera_calibration->GetCameraPtr();
        camera->SetFrameSize ( video_clip->GetFrameSize() );

        namedWindow ( camera->GetName(), CV_WINDOW_NORMAL );
        resizeWindow ( camera->GetName(), 800, 500 );
        moveWindow ( camera->GetName(), 800* ( i%2 ), 500* ( i/2 ) );
    }

    // Fetches frames and corners.
    unsigned valid_frame_count = 0;
    unsigned frame_index = 0;
    double current_time = 9.0;
    while ( true )
    {
        bool no_more_frame = true;
        for ( unsigned c=0; c<_synchronized_video_clips.size(); c++ )
        {
            VideoClip* video_clip = &_synchronized_video_clips[c];
            VideoCapture* video_capture = video_clip->GetVideoCapturePtr();
            CameraCalibration* camera_calibration = &_camera_calibrations[c];

            double local_time = current_time - video_clip->GetShiftInSeconds();
            // Skips if not started.
            if ( local_time < 0.0 )
            {
                continue;
            }
            video_capture->set ( CV_CAP_PROP_POS_MSEC, local_time * 1000.0 );
            // Skips if already ended.
            Mat frame;
            if ( !video_capture->read ( frame ) )
            {
                continue;
            }
            no_more_frame = false;

            Frame f;
            f.camera_name = video_clip->GetCameraName();
            f.global_index = frame_index;
            frame.copyTo ( f.original_frame );

            // Extracts corners.
            bool good_corners = camera_calibration->ExtractCornersAndSave ( &f );
            Utils::ShowFrameInWindow ( f.camera_name, f, good_corners, false );
            if ( good_corners )
            {
                cout << "\t" << f.camera_name << "-" << f.global_index << ": " << f.corner_count << " corners." << endl;
                valid_frame_count ++;
            }
        }
        current_time += _options.frame_gap;
        frame_index ++;

        char enter = cvWaitKey ( 1 );
        if ( no_more_frame || enter == 'q' )
        {
            break;
        }
    }

    cout << "... Done: " << valid_frame_count << " valid frames are found." << endl;
}

void CameraSystemCalibration::Calibrate()
{
    cout << endl << "Calibrating camera system ..." << endl;
    CalibrateMonoCameras();
    AlignCamerasAndBoards();
    cout << endl << "\tAverage reprojection error from shared frames is " << Reproject() << endl;
    OptimizeExtrinsics();
    double reprojection_error = Reproject();
    cout << endl << "\tAverage reprojection error from shared frames is " << reprojection_error << endl;

    if ( reprojection_error > 1.0 )
    {
        cout << endl << "\tAverage reprojection error is greater than 1.0 px. Calibration result may be not accurate enough." << endl;
    }
    else
    {
        _calibrated = true;
    }

    // Reposition all other cameras to the first camera transform, and print result to console.
    for ( unsigned i=0; i<_camera_calibrations.size(); i++ )
    {
	string camera_0_tag = "C"+_camera_calibrations[0].GetCameraName();
        string camera_i_tag = "C"+_camera_calibrations[i].GetCameraName();
	// Make C0 the camera system center, i,e. pt_c1 = C1 * C0.inv * pt_c0
	Mat repositioned_camera_i_transform = _vertex_pose_map[camera_i_tag] * _vertex_pose_map[camera_0_tag].inv();
        repositioned_camera_i_transform.copyTo ( _camera_extrinsics[i] );
	cout << endl << "Camera: " << _camera_calibrations[i].GetCameraName() << endl << _camera_extrinsics[i] << endl;
    }
}

void CameraSystemCalibration::CalibrateMonoCameras()
{
    cout << endl << "\tPerforming mono camera calibration ..." << endl;

    for ( unsigned c=0; c<_camera_calibrations.size(); c++ )
    {
        CameraCalibration* camera_calibration = &_camera_calibrations[c];
        cout << endl << "\t\tCalibrating camera [" << camera_calibration->GetCameraPtr()->GetName() << "] ..." << endl;

        camera_calibration->Calibrate();
        double reprojection_error = camera_calibration->Reproject();
        cout << "\t\tAverage reprojection error after initial calibration is " << reprojection_error << endl;
        do
        {
            camera_calibration->OptimizeFully();
            reprojection_error = camera_calibration->Reproject();
            cout << "\t\tAverage reprojection error after non-linear optimization is " << reprojection_error << endl;
        }
        while ( camera_calibration->RejectFrames ( 10, 1000, true ) > 0 );

        if ( reprojection_error > 1.0 )
        {
            cout << endl << "\t\tAverage reprojection error is greater than 1.0 px. Mono calibration result may be not accurate enough." << endl;
        }

        if ( _options.save_valid_frames )
        {
            camera_calibration->SaveAllValidFrames ( _options.save_frames_folder, true, true );
        }
    }

    cout << "\t... Done." << endl;
}

void CameraSystemCalibration::AlignCamerasAndBoards()
{
    cout << endl << "\tAligning cameras and valid frames initially ..." << endl;
    // Initializes the backward edges from frame to camera; forward edges are stored as valid frame in camera calibration.
    unordered_map<unsigned, vector<unsigned>> frame_to_camera;
    for ( unsigned i=0; i<_camera_calibrations.size(); i++ )
    {
        CameraCalibration* camera_calibration = &_camera_calibrations[i];
        vector<Frame>* valid_frames = camera_calibration->GetValidFramesPtr();
        for ( unsigned j=0; j<valid_frames->size(); j++ )
        {
            Frame* frame = & ( *valid_frames ) [j];
            frame_to_camera[frame->global_index].push_back ( i );
        }
    }
    // Determines initial poses using BFS starting from the first camera.
    bool is_camera_vertex = true;
    queue<unsigned> visiting_vertex;
    queue<Mat> visiting_pose;
    visiting_vertex.push ( 0 );
    visiting_pose.push ( Mat::eye ( 4, 4, CV_64FC1 ) );
    while ( !visiting_vertex.empty() )
    {
        unsigned level_size = visiting_vertex.size();
        unordered_set<unsigned> vertex_added_in_queue;
        for ( unsigned i=0; i<level_size; i++ )
        {
            if ( is_camera_vertex )
            {
                unsigned camera_index = visiting_vertex.front();
                visiting_vertex.pop();
                Mat camera_pose = visiting_pose.front();
                visiting_pose.pop();
                CameraCalibration* camera_calibration = &_camera_calibrations[camera_index];
                string camera_tag = "C"+camera_calibration->GetCameraName();
                _vertex_pose_map[camera_tag] = camera_pose;
                // Collects next level.
                vector<Frame>* valid_frames = camera_calibration->GetValidFramesPtr();
                for ( unsigned j=0; j<valid_frames->size(); j++ )
                {
                    Frame* frame = & ( *valid_frames ) [j];
                    string frame_tag = "F" + to_string ( frame->global_index );
                    if ( _vertex_pose_map.find ( frame_tag ) == _vertex_pose_map.end() &&
                            vertex_added_in_queue.find ( frame->global_index ) == vertex_added_in_queue.end() )
                    {
                        vertex_added_in_queue.insert ( frame->global_index );
                        visiting_vertex.push ( frame->global_index );
                        // The frame transform matrix F, so that pt_world = F * pt_frame.
                        visiting_pose.push ( camera_pose.inv() * Utils::GetTransform44From34 ( frame->transform ) );
                    }
                }
            }
            else
            {
                unsigned frame_global_index = visiting_vertex.front();
                visiting_vertex.pop();
                Mat frame_pose = visiting_pose.front();
                visiting_pose.pop();
                vector<unsigned> captured_cameras = frame_to_camera[frame_global_index];
                // Only keeps shared frames.
                if ( captured_cameras.size() > 0 ) // add all frames.
                {
                    string frame_tag = "F"+to_string ( frame_global_index );
                    _vertex_pose_map[frame_tag] = frame_pose;
                    // Collects next level.
                    for ( unsigned j=0; j<captured_cameras.size(); j++ )
                    {
                        unsigned camera_index = captured_cameras[j];
                        CameraCalibration* camera_calibration = &_camera_calibrations[camera_index];
                        string camera_tag = "C"+camera_calibration->GetCameraName();
                        if ( _vertex_pose_map.find ( camera_tag ) ==_vertex_pose_map.end() &&
                                vertex_added_in_queue.find ( camera_index ) == vertex_added_in_queue.end() )
                        {
                            Frame* frame = camera_calibration->GetFrameWithGloablIndex ( frame_global_index );
                            vertex_added_in_queue.insert ( camera_index );
                            visiting_vertex.push ( camera_index );
                            // The camera transform matrix C, so that pt_camera = C * pt_world.
                            visiting_pose.push ( Utils::GetTransform44From34 ( frame->transform ) * frame_pose.inv() );
                        }
                    }
                }
            }
        }
        is_camera_vertex = !is_camera_vertex;
    }
    // Checks connectivity: whether all cameras are covered.
    for ( unsigned i=0; i<_camera_calibrations.size(); i++ )
    {
        CameraCalibration* camera_calibration = &_camera_calibrations[i];
        string camera_tag = "C"+camera_calibration->GetCameraName();
        if ( _vertex_pose_map.find ( camera_tag ) == _vertex_pose_map.end() )
        {
            cerr << "Camera ["<<camera_calibration->GetCameraName() << "] doesn't share any valid frame with other cameras." << endl;
            exit ( -1 );
        }
    }

    cout << endl << "\t... Done." << endl;
}

void CameraSystemCalibration::OptimizeExtrinsics()
{
    cout << endl << "\tOptimizing extrinsic parameters of all known vertices (cameras and frames) ..." << endl;
    // Collects all intrinsics.
    unordered_map<string, Mat> all_rotation_vectors, all_translation_vectors;
    double all_intrinsics[_camera_calibrations.size()][TOTAL_SIZE];
    for ( unsigned i=0; i<_camera_calibrations.size(); i++ )
    {
        CameraCalibration* camera_calibration = &_camera_calibrations[i];
        camera_calibration->GetCameraPtr()->GetIntrinsicParameters ( all_intrinsics[i] );
    }
    // Collectws all extrinsics.
    for ( auto it=_vertex_pose_map.begin(); it!=_vertex_pose_map.end(); ++it )
    {
        Mat rotation_vector, translation_vector;
        Mat transform_4_4 = it->second;
        Utils::GetRAndTVectorsFromTransform ( transform_4_4.rowRange ( 0, 3 ), &rotation_vector, &translation_vector );
        all_rotation_vectors[it->first] = rotation_vector;
        all_translation_vectors[it->first] = translation_vector;
    }
    // Forms up the ceres problem.
    ceres::Problem problem;
    for ( unsigned i=0; i<_camera_calibrations.size(); i++ )
    {
        CameraCalibration* camera_calibration = &_camera_calibrations[i];
        string camera_tag = "C"+camera_calibration->GetCameraName();
        vector<Frame>* valid_frames = camera_calibration->GetValidFramesPtr();
        for ( unsigned j=0; j<valid_frames->size(); j++ )
        {
            Frame* frame = & ( *valid_frames ) [j];
            string frame_tag = "F"+to_string ( frame->global_index );
            if ( _vertex_pose_map.find ( frame_tag ) != _vertex_pose_map.end() )
            {
//                 for ( unsigned k=0; k<frame->corner_count; k++ )
//                 {
//                     Vec2d detected_corner = Vec2d ( frame->flatten_detected_corners_64.row ( k ) );
//                     Vec3d board_corner = Vec3d ( frame->flatten_board_corners_64.row ( k ) );
//                     ceres::CostFunction* cost_function = ErrorToOptimizeSystemExtrinsics::Create ( detected_corner, board_corner, all_intrinsics[i] );
//                     ceres::LossFunction* loss_function = new ceres::CauchyLoss ( 1.0 );
//                     problem.AddResidualBlock ( cost_function, loss_function,
//                                                ( double* ) all_rotation_vectors[camera_tag].data, ( double* ) all_translation_vectors[camera_tag].data,
//                                                ( double* ) all_rotation_vectors[frame_tag].data, ( double* ) all_translation_vectors[frame_tag].data );
//                 }

                for ( unsigned k=0; k<frame->corner_count; k++ )
                {
                    Vec2d detected_corner = Vec2d ( frame->flatten_detected_corners_64.row ( k ) );
                    Vec3d board_corner = Vec3d ( frame->flatten_board_corners_64.row ( k ) );
                    ceres::CostFunction* cost_function = ErrorToOptimizeSystemFully::Create ( detected_corner, board_corner );
                    ceres::LossFunction* loss_function = new ceres::CauchyLoss ( 1.0 );
                    problem.AddResidualBlock ( cost_function, loss_function, all_intrinsics[i],
                                               ( double* ) all_rotation_vectors[camera_tag].data, ( double* ) all_translation_vectors[camera_tag].data,
                                               ( double* ) all_rotation_vectors[frame_tag].data, ( double* ) all_translation_vectors[frame_tag].data );
                }

//                 ceres::CostFunction* cost_function = ErrorToOptimizeSystemExtrinsics2::Create ( frame->flatten_detected_corners_64,
// 												frame->flatten_board_corners_64, all_intrinsics[i] );
//                 ceres::LossFunction* loss_function = new ceres::CauchyLoss ( 1.0 );
//                 problem.AddResidualBlock ( cost_function, loss_function,
//                                            ( double* ) all_rotation_vectors[camera_tag].data, ( double* ) all_translation_vectors[camera_tag].data,
//                                            ( double* ) all_rotation_vectors[frame_tag].data, ( double* ) all_translation_vectors[frame_tag].data );
            }
        }
    }
    // Solves ceres problem.
    ceres::Solver::Options options;
    options.num_threads = 4;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = _ceres_details_enabled;
    ceres::Solver::Summary summary;
    Solve ( options, &problem, &summary );
    if ( _ceres_details_enabled )
    {
        cout << summary.FullReport() << endl;
    }
    // Updates poses of vertices.
    for ( auto it=_vertex_pose_map.begin(); it!=_vertex_pose_map.end(); ++it )
    {
        string vertex_tag = it->first;
        Mat transform = Mat::eye ( 3, 4, CV_64FC1 );
        Utils::GetTransformFromRAndTVectors ( all_rotation_vectors[vertex_tag], all_translation_vectors[vertex_tag], &transform );
        transform.copyTo ( _vertex_pose_map[vertex_tag].rowRange ( 0, 3 ) );
    }
    // Updates intrinsic parameters.
    for ( unsigned i=0; i<_camera_calibrations.size(); i++ )
    {
        CameraCalibration* camera_calibration = &_camera_calibrations[i];
        camera_calibration->GetCameraPtr()->SetIntrinsics ( all_intrinsics[i] );
    }

    cout << endl << "\t... Done." << endl;
}

double CameraSystemCalibration::Reproject()
{
    double error_sum = 0.0;
    int corners_count = 0;
    for ( unsigned i=0; i<_camera_calibrations.size(); i++ )
    {
        CameraCalibration* camera_calibration = &_camera_calibrations[i];
        string camera_tag = "C"+camera_calibration->GetCameraName();
        double intrinsics[TOTAL_SIZE];
        camera_calibration->GetCameraPtr()->GetIntrinsicParameters ( intrinsics );
        Mat camera_transform = _vertex_pose_map[camera_tag];

        vector<Frame>* valid_frames = camera_calibration->GetValidFramesPtr();
        for ( unsigned j=0; j<valid_frames->size(); j++ )
        {
            Frame* frame = & ( *valid_frames ) [j];
            string frame_tag = "F"+to_string ( frame->global_index );
            if ( _vertex_pose_map.find ( frame_tag ) != _vertex_pose_map.end() )
            {
                Mat frame_transform = _vertex_pose_map[frame_tag];
                Mat local_transform = camera_transform * frame_transform;
                Mat local_rotation_vector, local_translation_vector, reprojected_corners;
                Utils::GetRAndTVectorsFromTransform ( local_transform.rowRange ( 0, 3 ), &local_rotation_vector, &local_translation_vector );
                Utils::ReprojectCornersInFrame ( intrinsics, ( double* ) local_rotation_vector.data, ( double* ) local_translation_vector.data,
                                                 frame->flatten_board_corners_64, &reprojected_corners );
                // Accumulates reprojection error.
                Mat reprojection_error = reprojected_corners - frame->flatten_detected_corners_64;
                double error_in_frame = 0.0;
                for ( unsigned r=0; r<frame->corner_count; r++ )
                {
                    error_in_frame += norm ( reprojection_error.row ( r ) );
                }
                error_sum += error_in_frame;
                corners_count += frame->corner_count;
            }
        }
    }

    return error_sum / corners_count;
}

void CameraSystemCalibration::ReadVideoFileAndCameraNames ( const string& config_file_name, vector< string >* camera_names, vector< string >* video_file_names )
{
    FileStorage file_storage ( config_file_name, FileStorage::READ );
    FileNode video_file_node = file_storage["Videos"];
    FileNodeIterator video_file_iterator = video_file_node.begin();
    FileNodeIterator video_file_iterator_end = video_file_node.end();
    for ( ; video_file_iterator != video_file_iterator_end; ++video_file_iterator )
    {
        string video_file_name = ( string ) ( *video_file_iterator ) ["File"];
        string camera_name = ( string ) ( *video_file_iterator ) ["CameraName"];
        if ( !Utils::FileExists ( video_file_name ) )
        {
            exit ( -1 );
        }
        video_file_names->push_back ( video_file_name );
        camera_names->push_back ( camera_name );
    }
    file_storage.release();
}

void CameraSystemCalibration::ReadCalibrationParameters ( const string& config_file_name, CameraSystemCalibrationOptions* options )
{
    FileStorage file_storage ( config_file_name, FileStorage::READ );
    if ( file_storage["FrameFolder"].empty() || !Utils::FileExists ( file_storage["FrameFolder"] ) )
    {
        options->save_valid_frames = false;
    }
    else
    {
        options->save_valid_frames = true;
        options->save_frames_folder = ( string ) file_storage["FrameFolder"];
    }
    options->frame_gap = file_storage["FrameGap"];
    options->dictionary = aruco::getPredefinedDictionary ( file_storage["ArucoDictionary"] );
    options->charuco_board = aruco::CharucoBoard::create ( file_storage["BoardSize.x"], file_storage["BoardSize.y"], file_storage["SquareSize"], file_storage["MarkerSize"], options->dictionary );
    options->detector_parameters = aruco::DetectorParameters::create();
    file_storage.release();
}

void CameraSystemCalibration::ReadArucoParameters ( const string& aruco_file_name, CameraSystemCalibrationOptions* options )
{
    if ( options->detector_parameters == nullptr )
    {
        options->detector_parameters = aruco::DetectorParameters::create();
    }
    FileStorage file_storage ( aruco_file_name, FileStorage::READ );
    file_storage["adaptiveThreshWinSizeMin"] >> options->detector_parameters->adaptiveThreshWinSizeMin;
    file_storage["adaptiveThreshWinSizeMax"] >> options->detector_parameters->adaptiveThreshWinSizeMax;
    file_storage["adaptiveThreshWinSizeStep"] >> options->detector_parameters->adaptiveThreshWinSizeStep;
    file_storage["adaptiveThreshConstant"] >> options->detector_parameters->adaptiveThreshConstant;
    file_storage["minMarkerPerimeterRate"] >> options->detector_parameters->minMarkerPerimeterRate;
    file_storage["maxMarkerPerimeterRate"] >> options->detector_parameters->maxMarkerPerimeterRate;
    file_storage["polygonalApproxAccuracyRate"] >> options->detector_parameters->polygonalApproxAccuracyRate;
    file_storage["minCornerDistanceRate"] >> options->detector_parameters->minCornerDistanceRate;
    file_storage["minDistanceToBorder"] >> options->detector_parameters->minDistanceToBorder;
    file_storage["minMarkerDistanceRate"] >> options->detector_parameters->minMarkerDistanceRate;
    file_storage["cornerRefinementWinSize"] >> options->detector_parameters->cornerRefinementWinSize;
    file_storage["cornerRefinementMaxIterations"] >> options->detector_parameters->cornerRefinementMaxIterations;
    file_storage["cornerRefinementMinAccuracy"] >> options->detector_parameters->cornerRefinementMinAccuracy;
    file_storage["markerBorderBits"] >> options->detector_parameters->markerBorderBits;
    file_storage["perspectiveRemovePixelPerCell"] >> options->detector_parameters->perspectiveRemovePixelPerCell;
    file_storage["perspectiveRemoveIgnoredMarginPerCell"] >> options->detector_parameters->perspectiveRemoveIgnoredMarginPerCell;
    file_storage["maxErroneousBitsInBorderRate"] >> options->detector_parameters->maxErroneousBitsInBorderRate;
    file_storage["minOtsuStdDev"] >> options->detector_parameters->minOtsuStdDev;
    file_storage["errorCorrectionRate"] >> options->detector_parameters->errorCorrectionRate;
}

void CameraSystemCalibration::SaveResults()
{
    FileStorage file_storage ( _output_file_name, FileStorage::WRITE );
    file_storage << "Cameras" << "[";
    for ( unsigned i=0; i<_camera_calibrations.size(); i++ )
    {
        Camera* camera = _camera_calibrations[i].GetCameraPtr();
        double intrinsics[TOTAL_SIZE];
        camera->GetIntrinsicParameters ( intrinsics );
        Mat rotation_vector, translation_vector;
        Utils::GetRAndTVectorsFromTransform ( _camera_extrinsics[i].rowRange(0, 3), &rotation_vector, &translation_vector );

        file_storage << "{";
	file_storage << "Name" << camera->GetName();
        // Writes affine parameters.
        file_storage << "C" << intrinsics[0] << "D" << intrinsics[1] << "E" << intrinsics[2];
        file_storage << "U0" << intrinsics[3] << "V0" << intrinsics[4];
        // Writes poly parameters.
        file_storage << "Poly" << "[:";
        for ( unsigned j=0; j<POLY_SIZE; j++ )
        {
            file_storage << intrinsics[POLY_START+j];
        }
        file_storage << "]";
        // Writes inverse poly parameters.
        file_storage << "InversePoly" << "[:";
        for ( unsigned j=0; j<INV_POLY_SIZE; j++ )
        {
            file_storage << intrinsics[INV_POLY_START+j];
        }
        file_storage << "]";
        // Writes extrinsic parameters.
	file_storage.writeComment("Extrinsic parameters: [0-2] rotation vector, [3-5] translation vector.", true);
        file_storage << "Extrinsic" << "[:";
        for ( unsigned j=0; j<3; j++ )
        {
            file_storage << rotation_vector.at<double> ( j, 0 );
        }
        for ( unsigned j=0; j<3; j++ )
        {
            file_storage << translation_vector.at<double> ( j, 0 );
        }
        file_storage << "]";
        file_storage << "}";
    }
    file_storage << "]";
    file_storage.release();
}

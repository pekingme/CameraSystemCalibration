#include "camera_system_calibration.h"
#include "video_synchronizer.h"
#include "utils.h"

CameraSystemCalibration::CameraSystemCalibration ( const string& output_file_name, const string& config_file_name, const string& detector_file_name )
    : _output_file_name ( output_file_name )
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
        _camera_calibrations.push_back ( CameraCalibration ( _synchronized_video_clips[i].GetCameraName(), _options ) );
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
    double current_time = 0.0;
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
    OptimizeExtrinsics();
    Reproject();
}

void CameraSystemCalibration::CalibrateMonoCameras()
{

}

void CameraSystemCalibration::AlignCamerasAndBoards()
{
// TODO
}

void CameraSystemCalibration::OptimizeExtrinsics()
{
// TODO
}

void CameraSystemCalibration::Reproject()
{
// TODO
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

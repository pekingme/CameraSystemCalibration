#include "camera_system_calibration.h"
#include "video_synchronizer.h"
#include "utils.h"

CameraSystemCalibration::CameraSystemCalibration ( const string& output_file_name, const string& config_file_name, const string& detector_file_name )
{
    cout << "Loading camera calibration system configurations ..." << endl;

    // Checks if input files exist.
    if ( !Utils::fileExists ( config_file_name ) || !Utils::fileExists ( detector_file_name ) ) exit ( -1 );
    cout << "\tCalibration configuration file: " << config_file_name << endl;
    cout << "\tArUco detector setting file: " << detector_file_name << endl;
    cout << "\tCalibration result file: " << output_file_name << endl;

    // Gets names of all video files and checks all files exist.
    vector<string> video_file_names = GetVideoFileNames(config_file_name);
    
    // Synchronizes all videos.
    FileStorage file_storage (config_file_name, FileStorage::READ);
    double shift_window = file_storage["MaxShift"];
    VideoSynchronizer video_synchronizer;
    video_synchronizer.LoadVideosWithFileNames(video_file_names);
    video_synchronizer.SynchronizeVideoWithAudio(shift_window*2);
    
}

CameraSystemCalibration::~CameraSystemCalibration(){}

vector<string> CameraSystemCalibration::GetVideoFileNames ( const string& config_file_name )
{
    FileStorage file_storage ( config_file_name, FileStorage::READ );
    vector<string> video_file_vector;
    FileNode video_file_node = file_storage["Videos"];
    FileNodeIterator video_file_iterator = video_file_node.begin();
    FileNodeIterator video_file_iterator_end = video_file_node.end();
    for ( ; video_file_iterator != video_file_iterator_end; ++video_file_iterator )
    {
        string video_file_name = ( string ) *video_file_iterator;
        if ( Utils::fileExists ( video_file_name ) ) exit ( -1 );
        video_file_vector.push_back ( video_file_name );
    }
    file_storage.release();
    
    return video_file_vector;
}


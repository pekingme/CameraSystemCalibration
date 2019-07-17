#include "video_synchronizer.h"

void VideoSynchronizer::LoadVideosWithFileNames ( const vector<string>& video_file_names, const vector<string>& camera_names )
{
    if ( video_file_names.size() != camera_names.size() )
    {
        cerr << "The numbers of camera names and video names don't match." << endl;
        exit ( -1 );
    }
    if ( video_file_names.size() == 0 )
    {
        cerr << "No videos are associated for calibration." << endl;
        exit ( -1 );
    }
    _video_clip_vector.clear();
    for ( unsigned i=0; i<video_file_names.size(); i++ )
    {
        _video_clip_vector.push_back ( VideoClip ( video_file_names[i], camera_names[i] ) );
    }
}

void VideoSynchronizer::SynchronizeVideoWithAudio ( const int shift_window )
{
    if ( _video_clip_vector.size() < 2 )
    {
        cerr << "Synchronization can only be performed between 2 or more videos." << endl;
        exit ( -1 );
    }

    // Load audio samples from all videos
    vector<Mat> audio_samples;
    for ( unsigned i=0; i<_video_clip_vector.size(); i++ )
    {
        Mat samples;
        VideoClip* video_clip = &_video_clip_vector[i];
        if ( !video_clip->ExtractAudioSamples ( &samples, shift_window ) )
        {
            cerr << "Cannot read audio samples from video " << i << endl;
            exit ( -1 );
        }
        audio_samples.push_back ( samples );
    }

    // Calculates time shifts relative to the first video.
    _video_clip_vector[0].SetShiftInSeconds ( 0.0 );
    double max_leading = 0.0;
    for ( unsigned i=1; i<_video_clip_vector.size(); i++ )
    {
        double leading = SynchronizeToReference ( audio_samples[0], audio_samples[i], &_video_clip_vector[i] );
        max_leading = min ( max_leading, leading );
    }

    // Offsets all time shift relative to the first started video.
    for ( unsigned i=0; i<_video_clip_vector.size(); i++ )
    {
        VideoClip* video_clip = &_video_clip_vector[i];
        video_clip->SetShiftInSeconds ( video_clip->GetShiftInSeconds() - max_leading );
        cout << "\t" << video_clip->GetShiftInSeconds() << " seconds: " << video_clip->GetFileName() << endl;
    }
}

void VideoSynchronizer::ViewSynchronizedVideos()
{
    if ( _video_clip_vector.size() < 1 )
    {
        cerr << "No video to playback." << endl;
        exit ( -1 );
    }
    // Loads all video files and creates windows.
    for ( unsigned i=0; i<_video_clip_vector.size(); i++ )
    {
        VideoClip* video_clip = &_video_clip_vector[i];
        if ( !video_clip->IsLoaded() )
        {
            video_clip->LoadVideoCapture();
        }
        else
        {
            VideoCapture* video_capture = video_clip->GetVideoCapturePtr();
            video_capture->set ( CV_CAP_PROP_POS_FRAMES, 0 );
        }
        namedWindow ( video_clip->GetCameraName(), CV_WINDOW_NORMAL );
    }

    // Playback.
    cout << "Press any key to start playback, then 'q' to quit." << endl;
    cvWaitKey ( 0 );
    double current_time = 0.0;
    while ( true )
    {
        bool no_more_frame = true;
        double next_frame_time = current_time;
        for ( unsigned i=0; i<_video_clip_vector.size(); i++ )
        {
            VideoClip* video_clip = &_video_clip_vector[i];
            double video_local_time = current_time - video_clip->GetShiftInSeconds();
            VideoCapture* video_capture = video_clip->GetVideoCapturePtr();
            // Skip if not started.
            if ( video_local_time < 0.0 )
            {
                continue;
            }
            video_capture->set ( CV_CAP_PROP_POS_MSEC, video_local_time * 1000.0 );
            // Skip if finished.
            Mat frame;
            if ( !video_capture->read ( frame ) )
            {
                continue;
            }

            no_more_frame = false;
            imshow ( video_clip->GetFileName(), frame );
            next_frame_time = max ( next_frame_time, video_capture->get ( CV_CAP_PROP_POS_MSEC ) / 1000.0 + video_clip->GetShiftInSeconds() );
        }
        char enter = cvWaitKey ( 1 );
        if ( enter == 'q' || no_more_frame )
        {
            break;
        }
        else
        {
            current_time = next_frame_time;
        }
    }
}

double VideoSynchronizer::SynchronizeToReference ( const Mat& reference_samples, const Mat& audio_samples, VideoClip* video_clip )
{
    int padding_size = audio_samples.cols;
    Mat reference_audio_sample_with_padding, correlation;
    copyMakeBorder ( reference_samples, reference_audio_sample_with_padding, 0, 0, padding_size, padding_size, cv::BORDER_CONSTANT, Scalar ( 0 ) );
    matchTemplate ( reference_audio_sample_with_padding, audio_samples, correlation, cv::TM_CCORR );

    double min_val, max_val;
    Point min_pos, max_pos;
    minMaxLoc ( correlation, &min_val, &max_val, &min_pos, &max_pos, Mat() );
    video_clip->SetShiftInSeconds ( ( max_pos.x-padding_size ) / video_clip->GetAudioSampleRate() );

    return video_clip->GetShiftInSeconds();
}


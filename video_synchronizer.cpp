#include "video_synchronizer.h"

void VideoSynchronizer::LoadVideosWithFileNames ( const vector<string>& video_file_names )
{
    if ( video_file_names.size() == 0 )
    {
        cerr << "No videos are associated for calibration." << endl;
        exit ( -1 );
    }
    _video_clip_vector.clear();
    for ( auto it=video_file_names.begin(); it!=video_file_names.end(); ++it )
    {
        _video_clip_vector.push_back ( VideoClip ( ( string ) *it ) );
    }
}

bool VideoSynchronizer::SynchronizeVideoWithAudio ( const int shift_window )
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
        VideoClip video_clip = _video_clip_vector[i];
        if ( !video_clip.ExtractAudioSamples ( &samples, shift_window ) )
        {
            cerr << "Cannot read audio samples from video " << i << endl;
            exit ( -1 );
        }
        audio_samples.push_back ( samples );
    }

    // Calculates time shifts relative to the first video.
    VideoClip reference_clip = _video_clip_vector[0];
    reference_clip.SetShiftInSeconds ( 0.0 );
    Mat reference_audio_sample = audio_samples[0];
    double max_leading = 0.0;
    for ( unsigned i=1; i<_video_clip_vector.size(); i++ )
    {
        VideoClip video_clip = _video_clip_vector[i];
        Mat audio_sample = audio_samples[i];
        int padding_size = audio_sample.rows;
        Mat reference_audio_sample_with_padding, correlation;
        copyMakeBorder ( reference_audio_sample, reference_audio_sample_with_padding, 0, 0, padding_size, padding_size, cv::BORDER_CONSTANT, Scalar ( 0 ) );
        matchTemplate ( reference_audio_sample_with_padding, audio_sample, correlation, cv::TM_CCORR );

        double min_val, max_val;
        Point min_pos, max_pos;
        minMaxLoc ( correlation, &min_val, &max_val, &min_pos, &max_pos, Mat() );
        video_clip.SetShiftInSeconds ( ( max_pos.x-padding_size ) / video_clip.GetAudioSampleRate() );
        max_leading = min ( max_leading, video_clip.GetShiftInSeconds() );
    }

    // Offsets all time shift relative to the first started video.
    if ( max_leading < 0.0 )
    {
        for ( unsigned i=0; i<_video_clip_vector.size(); i++ )
        {
            VideoClip video_clip = _video_clip_vector[i];
            video_clip.SetShiftInSeconds ( video_clip.GetShiftInSeconds()-max_leading );
        }
    }

    return true;
}


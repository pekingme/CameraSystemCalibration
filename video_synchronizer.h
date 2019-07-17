#ifndef VIDEOSYNCHRONIZER_H
#define VIDEOSYNCHRONIZER_H

#include <iostream>
#include <string>
#include <vector>
#include "video_clip.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class VideoSynchronizer
{
public:
    // Creates video clips from the filenames with camera names.
    void LoadVideosWithFileNames ( const vector<string>& video_file_names, const vector<string>& camera_names );

    // Synchronizes video clips based on audio samples in a certain range of time.
    // The maximum time shift between the first and last video clips is no more than the window.
    void SynchronizeVideoWithAudio ( const int shift_window );

    // Playbacks all videos with synchronizing shifts together.
    void ViewSynchronizedVideos ();

    // Getter
    vector<VideoClip> GetVideoClips() {
        return _video_clip_vector;
    }
private:
    // Synchronizes one video clip by comparing its audio samples with referecne samples, and returns the amount of time it leads the reference.
    double SynchronizeToReference ( const Mat& reference_samples, const Mat& audio_samples, VideoClip* video_clip );

    vector<VideoClip> _video_clip_vector;
};

#endif // VIDEOSYNCHRONIZER_H

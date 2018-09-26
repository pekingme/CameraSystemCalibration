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
    // Creates video clips from the filenames.
    void LoadVideosWithFileNames ( const vector<string>& video_file_names );

    // Synchronizes video clips based on audio samples in a certain range of time.
    // The maximum time shift between the first and last video clips is no more than the window.
    bool SynchronizeVideoWithAudio ( const int shift_window );
private:
    vector<VideoClip> _video_clip_vector;
};

#endif // VIDEOSYNCHRONIZER_H

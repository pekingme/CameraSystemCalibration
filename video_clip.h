#ifndef VIDEOCLIP_H
#define VIDEOCLIP_H

#include <string>
#include "opencv2/opencv.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
};

using namespace std;
using namespace cv;

class VideoClip
{
public:
    VideoClip ( const string& file_name, const string& camera_name )
        : _file_name ( file_name ), _camera_name ( camera_name ), _loaded ( false ) {}
    ~VideoClip() {}

    bool ExtractAudioSamples ( Mat* mat, const int duration );
    void LoadVideoCapture();

    // Setters
    void SetShiftInSeconds ( double shift ) {
        _shift_in_seconds = shift;
    }
    // Getters
    string GetFileName() {
        return _file_name;
    }
    string GetCameraName() {
        return _camera_name;
    }
    double GetShiftInSeconds() {
        return _shift_in_seconds;
    }
    double GetAudioSampleRate() {
        return _audio_sample_rate;
    }
    Size GetFrameSize() {
        return _frame_size;
    }
    bool IsLoaded() {
        return _loaded;
    }
    VideoCapture* GetVideoCapturePtr() {
        return &_video_capture;
    }

private:
    void CopySamplesToVector ( const AVCodecContext* codec_context, const AVFrame* frame, vector<float>& samples );

    string _file_name;
    string _camera_name;
    double _shift_in_seconds;
    double _audio_sample_rate;
    Size _frame_size;
    bool _loaded;
    VideoCapture _video_capture;
};

#endif // VIDEOCLIP_H

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
    VideoClip () {}
    VideoClip ( const string& file_name ) : _file_name ( file_name ) {}
    ~VideoClip() {}

    bool ExtractAudioSamples ( Mat* mat, const int duration );

    void SetShiftInSeconds ( const double shift ) {
        _shift_in_seconds = shift;
    }
    double GetShiftInSeconds() {
        return _shift_in_seconds;
    }
    double GetAudioSampleRate() {
        return _audio_sample_rate;
    }

private:
    void CopySamplesToVector ( const AVCodecContext* codec_context, const AVFrame* frame, vector<float>& samples );

    string _file_name;
    double _shift_in_seconds;
    double _audio_sample_rate;
};

#endif // VIDEOCLIP_H

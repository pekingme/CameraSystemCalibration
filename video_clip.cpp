#include "video_clip.h"

bool VideoClip::ExtractAudioSamples ( Mat* mat, const int duration )
{
    // Initializes FFmpeg.
    av_register_all();

    AVFrame* frame = av_frame_alloc();
    if ( !frame )
    {
        cerr << "FFmpeg: Fail to allocate AVFrame." << endl;
        exit ( -1 );
    }

    AVFormatContext* format_context = NULL;
    if ( avformat_open_input ( &format_context, _file_name.c_str(), NULL, NULL ) != 0 )
    {
        av_free ( frame );
        cerr << "FFmpeg: Fail to open file " << _file_name << endl;
        exit ( -1 );
    }

    // Finds the audio stream.
    AVCodec* codec = NULL;
    int stream_index = av_find_best_stream ( format_context, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0 );
    if ( stream_index < 0 )
    {
        av_free ( frame );
        avformat_close_input ( &format_context );
        cerr << "FFmpeg: Cannot find any audio stream in the file " << _file_name << endl;
        exit ( -1 );
    }

    AVStream* audio_stream = format_context->streams[stream_index];
    AVCodecContext* audio_codec_context = audio_stream->codec;
    audio_codec_context->codec = codec;

    if ( avcodec_open2 ( audio_codec_context, audio_codec_context->codec, NULL ) != 0 )
    {
        av_free ( frame );
        avformat_close_input ( &format_context );
        cerr << "FFmpeg: Cannot open the context with the decoder" << endl;
        exit ( -1 );
    }

    int sample_rate = audio_codec_context->sample_rate;
    _audio_sample_rate = sample_rate;
    int sample_size = av_get_bytes_per_sample ( audio_codec_context->sample_fmt );
    int duration_in_sample = duration * sample_rate;

    vector<float> samples;

    AVPacket read_packet;
    av_init_packet ( &read_packet );

    // Reads the packets in a loop.
    while ( av_read_frame ( format_context, &read_packet ) == 0 )
    {
        if ( read_packet.stream_index == audio_stream->index )
        {
            AVPacket decoding_packet = read_packet;
            bool enough_samples = false;
            // Audio packets can have multiple audio frames in a single packet
            while ( decoding_packet.size > 0 )
            {
                // Tries to decode the packet into a frame.
                // Some frames reply on multiple packets, so we have to make sure the frame is finished before we can use it.
                int got_frame = 0;
                int result = avcodec_decode_audio4 ( audio_codec_context, frame, &got_frame, &decoding_packet );

                if ( result >= 0 && got_frame )
                {
                    decoding_packet.size -= result;
                    decoding_packet.data += result;

                    // We now have a fully decoded audio frame
                    CopySamplesToVector ( audio_codec_context, frame, samples );
                }
                else
                {
                    decoding_packet.size = 0;
                    decoding_packet.data = NULL;
                }

                if ( samples.size() >= ( unsigned ) duration_in_sample )
                {
                    enough_samples = true;
                    break;
                }
            }

            if ( enough_samples )
            {
                break;
            }
        }
    }

    // av_free_packet() must be called after each call to av_read_frame() or memory will leak.
    av_free_packet ( &read_packet );

    // Some codecs will cause frames to be buffered up in the decoding process.
    // If the CODEC_CAP_DELAY flag is set, there can be buffered up frames that need to be flushed.
    // So we'll do that here
    if ( audio_codec_context->codec->capabilities & CODEC_CAP_DELAY )
    {
        av_init_packet ( &read_packet );
        // Decode all the remaining frames in the buffer, until the end is reached
        int got_frame = 0;
        while ( avcodec_decode_audio4 ( audio_codec_context, frame, &got_frame, &read_packet ) >= 0 && got_frame )
        {
            // We now have a fully decoded audio frame
            CopySamplesToVector ( audio_codec_context, frame, samples );
        }
    }

    // Cleans everything.
    av_free ( frame );
    avcodec_close ( audio_codec_context );
    avformat_close_input ( &format_context );

    // Copy vector to mat
    *mat = cv::Mat::zeros ( 1, samples.size(), CV_32FC1 );
    memcpy ( mat->data, samples.data(), samples.size() * sample_size );

    return samples.size() != 0;
}

void VideoClip::LoadVideoCapture()
{
    if ( &_video_capture != NULL && _video_capture.isOpened() )
    {
        _video_capture.set ( CV_CAP_PROP_POS_FRAMES, 0 );
        return;
    }
    _video_capture = VideoCapture ( _file_name );
    if ( !_video_capture.isOpened() )
    {
        cerr << "Cannot open video file: " << _file_name << endl;
        _loaded = false;
    }
    _frame_size = Size ( _video_capture.get ( CV_CAP_PROP_FRAME_WIDTH ), _video_capture.get ( CV_CAP_PROP_FRAME_HEIGHT ) );
    _loaded = true;
}

void VideoClip::CopySamplesToVector ( const AVCodecContext* codec_context, const AVFrame* frame, vector< float >& samples )
{
    float* data_begin = reinterpret_cast<float*> ( frame->data[0] );
    ulong size = frame->nb_samples * av_get_bytes_per_sample ( codec_context->sample_fmt ) / sizeof ( float );
    samples.insert ( samples.end(), data_begin, data_begin + size );
}


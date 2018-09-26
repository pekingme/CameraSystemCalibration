#include <string>
#include "opencv2/opencv.hpp"
#include "camera_system_calibration.h"

using namespace std;
using namespace cv;

namespace
{
const char* about = "This program calibrates a camera system using videos of ChArUco board\n";
const char* keys =
    "{help h ?||Print help message}"
    "{@output|<none>|Output file storing calibration results}"
    "{@config|<none>|Input file storing calibration configurations}"
    "{@aruco|<none>|Input file storing ArUco detector parameters}";
}

int main ( int argc, char** argv )
{
    CommandLineParser parser ( argc, argv, keys );
    parser.about ( about );

    if ( parser.has ( "help" ) || argc < 3 )
    {
        parser.printMessage();
        return 0;
    }

    string output_file_name = parser.get<string> ( "@output" );
    string config_file_name = parser.get<string> ( "@config" );
    string detector_file_name = parser.get<string> ( "@aruco" );

    if ( !parser.check() )
    {
        parser.printErrors();
        return 0;
    }

    CameraSystemCalibration camera_system_calibration (output_file_name, config_file_name, detector_file_name);
    camera_system_calibration.FetchCornersInFrames();
    camera_system_calibration.Calibrate();
    
}


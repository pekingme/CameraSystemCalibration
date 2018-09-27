/*
 * camera holds intrinsic and extrinsic parameters of the Scaramuzza's
 * omnidirectional camera model.
 */

#ifndef CAMERA_H
#define CAMERA_H

#include <string>
#include "opencv2/opencv.hpp"

#define AFFINE_SIZE 5
#define POLY_SIZE 5
#define INV_POLY_SIZE 12
#define TOTAL_SIZE (AFFINE_SIZE+POLY_SIZE+INV_POLY_SIZE)

using namespace std;
using namespace cv;

class Camera
{
public:
    Camera ( string name ) : _name ( name ) {};
    ~Camera() {};

    void SetFrameSize ( const Size& frame_size ) {
        _width = frame_size.width;
        _height = frame_size.height;
    }

    // Getters
    string GetName() {
        return _name;
    }
    int GetWidth() {
        return _width;
    }
    int GetHeight() {
        return _height;
    }
    double GetU0() {
        return _intrinsics[3];
    }
    double GetV0() {
        return _intrinsics[4];
    }

    // Setters
    void SetIntrinsics ( const double intrinsics[] ) {
        copy ( intrinsics, intrinsics+TOTAL_SIZE, _intrinsics );
    }
private:
    // Camera's name
    string _name;
    // Frame height
    int _height;
    // Frame width
    int _width;
    // Intrinsic parameters in format of [c, d, e, u0, v0, poly(5), inv_poly(12)]
    double _intrinsics[];
};

#endif // CAMERA_H

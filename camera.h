/*
 * camera holds intrinsic and extrinsic parameters of the Scaramuzza's
 * omnidirectional camera model.
 */

#ifndef CAMERA_H
#define CAMERA_H

#include <string>
#include "opencv2/opencv.hpp"

#define AFFINE_SIZE 5
#define POLY_START 5
#define POLY_SIZE 5
#define INV_POLY_START 10
#define INV_POLY_SIZE 12
#define TOTAL_SIZE (AFFINE_SIZE+POLY_SIZE+INV_POLY_SIZE)

using namespace std;
using namespace cv;

class Camera
{
public:
    Camera ( string name ) : _name ( name ) {};
    ~Camera() {};

    // Prints all camera parameters in console.
    void PrintCameraParameters ( const string& prefix );

    // Getters
    string GetName() {
        return _name;
    }
    Vec2i GetFrameSize(){
      return Vec2i(_width, _height);
    }
    Vec2d GetCenter() {
        return Vec2d ( _intrinsics[3], _intrinsics[4] );
    }
    void GetPolyParameters(double* poly_parameters) {
        copy ( _intrinsics+POLY_START, _intrinsics+POLY_START+POLY_SIZE, poly_parameters );
    }
    void GetInversePolyParameters(double* inverse_poly_parameters) {
        copy ( _intrinsics+INV_POLY_START, _intrinsics+INV_POLY_START+INV_POLY_SIZE, inverse_poly_parameters );
    }

    // Setters
    void SetPolyParameters ( double* values ) {
        copy ( values, values+POLY_SIZE, _intrinsics+POLY_START );
    }
    void SetInversePolyParameters ( double* values ) {
        copy ( values, values+INV_POLY_SIZE, _intrinsics+INV_POLY_START );
    }
    void SetIntrinsics ( const double intrinsics[] ) {
        copy ( intrinsics, intrinsics+TOTAL_SIZE, _intrinsics );
    }

    void SetFrameSize ( const Size& frame_size ) {
        _width = frame_size.width;
        _height = frame_size.height;
    }
private:
    // Camera's name
    string _name;
    // Frame height
    int _height;
    // Frame width
    int _width;
    // Intrinsic parameters in format of [c, d, e, u0, v0, poly(5), inv_poly(12)]
    double _intrinsics[TOTAL_SIZE];
};

#endif // CAMERA_H

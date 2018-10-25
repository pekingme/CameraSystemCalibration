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
    Camera () {}
    Camera ( const string& name ) : _name ( name ) {}
    Camera ( const string& name, const double c, const double d, const double e, const double u0, const double v0, const double* poly, const double* inv_poly )
        : _name ( name )  {
        _intrinsics[0] = c;
        _intrinsics[1] = d;
        _intrinsics[2] = e;
        _intrinsics[3] = u0;
        _intrinsics[4] = v0;
        SetPolyParameters ( poly );
        SetInversePolyParameters ( inv_poly );
    }

    ~Camera() {}

    // Prints all camera parameters in console.
    void PrintCameraParameters ( const string& prefix );

    // Getters
    string GetName() const {
        return _name;
    }
    Vec2i GetFrameSize() const {
        return Vec2i ( _width, _height );
    }
    Vec2d GetCenter() const {
        return Vec2d ( _intrinsics[3], _intrinsics[4] );
    }
    void GetPolyParameters ( double* poly_parameters ) const {
        copy ( _intrinsics+POLY_START, _intrinsics+POLY_START+POLY_SIZE, poly_parameters );
    }
    void GetInversePolyParameters ( double* inverse_poly_parameters ) const {
        copy ( _intrinsics+INV_POLY_START, _intrinsics+INV_POLY_START+INV_POLY_SIZE, inverse_poly_parameters );
    }
    void GetIntrinsicParameters ( double* intrinsic_parameters ) const {
        copy ( _intrinsics, _intrinsics+TOTAL_SIZE, intrinsic_parameters );
    }

    // Setters
    void SetPolyParameters ( const double* values ) {
        copy ( values, values+POLY_SIZE, _intrinsics+POLY_START );
    }
    void SetInversePolyParameters ( const double* values ) {
        copy ( values, values+INV_POLY_SIZE, _intrinsics+INV_POLY_START );
    }
    void SetIntrinsicsParameters ( const double* intrinsics ) {
        copy ( intrinsics, intrinsics+TOTAL_SIZE, _intrinsics );
    }
    void CopyIntrinsicsFrom( const Camera& cam );

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

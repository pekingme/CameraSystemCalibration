/*
 * camera holds intrinsic and extrinsic parameters of the Scaramuzza's
 * omnidirectional camera model.
 */

#ifndef CAMERA_H
#define CAMERA_H

#include <string>

#define AFFINE_SIZE 5
#define POLY_SIZE 5
#define INV_POLY_SIZE 12
#define TOTAL_SIZE (AFFINE_SIZE+POLY_SIZE+INV_POLY_SIZE)

using namespace std;

class camera
{
public:
    camera ( string name ) : _name ( name ) {};
    ~camera() {};
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

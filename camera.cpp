#include "camera.h"

void Camera::PrintCameraParameters ( const string& prefix )
{
    cout << prefix << "Camera [" << _name << "]:" << endl;
    cout << prefix << "\tCenter:" << endl;
    cout << prefix << "\t\tu0 = " << _intrinsics[3] << endl;
    cout << prefix << "\t\tv0 = " << _intrinsics[4] << endl;
    cout << prefix << "\tAffine:" << endl;
    cout << prefix << "\t\tc = " << _intrinsics[0] << endl;
    cout << prefix << "\t\td = " << _intrinsics[1] << endl;
    cout << prefix << "\t\te = " << _intrinsics[2] << endl;
    cout << prefix << "\tPoly parameters:" << endl;
    for ( int i=0; i<POLY_SIZE; i++ )
    {
        cout << prefix << "\t\tp[" << i << "] = " << _intrinsics[5+i] << endl;
    }
    cout << prefix << "\tInverse poly parameters:" << endl;
    for ( int i=0; i<INV_POLY_SIZE; i++ )
    {
        cout << prefix << "\t\tinv_p[" << i << "] = " << _intrinsics[10+i] << endl;
    }
}

void Camera::CopyIntrinsicsFrom ( const Camera& cam )
{
    double intrinsics[TOTAL_SIZE];
    cam.GetIntrinsicParameters ( intrinsics );
    SetIntrinsicsParameters ( intrinsics );
}

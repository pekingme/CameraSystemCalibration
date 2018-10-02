#ifndef OPTIMIZATIONSTRUCTS_H
#define OPTIMIZATIONSTRUCTS_H

#include <vector>
#include "camera.h"
#include "utils.h"
#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"

using namespace std;
using namespace cv;

struct ErrorToSolvePolyAndT3 {
public:
    ErrorToSolvePolyAndT3 ( const Vec2d& detected_corner, const Vec3d& board_corner, const Mat& transform )
        : _detected_corner ( detected_corner ), _board_corner ( board_corner ), _transform ( transform ) {
        double rho = norm ( detected_corner );
        _rho_powers.push_back ( 1 );
        for ( int i=1; i<=4; i++ ) {
            _rho_powers.push_back ( _rho_powers[i-1] * rho );
        }
        _a = transform.at<double> ( 1,0 ) *board_corner[0]+transform.at<double> ( 1,1 ) *board_corner[1]+transform.at<double> ( 1,3 );
        _c = transform.at<double> ( 0,0 ) *board_corner[0]+transform.at<double> ( 0,1 ) *board_corner[1]+transform.at<double> ( 0,3 );
        _b = detected_corner[1] * ( transform.at<double> ( 2,0 ) *board_corner[0]+transform.at<double> ( 2,1 ) *board_corner[1] );
        _d = detected_corner[0] * ( transform.at<double> ( 2,0 ) *board_corner[0]+transform.at<double> ( 2,1 ) *board_corner[1] );
    }

    bool operator() ( const double* poly, const double* t3, double* residuals ) const {
        double constraint_1 = poly[2] * 2 * _rho_powers[1] + poly[3] * 3 * _rho_powers[2] + poly[4] * 4 * _rho_powers[3];
        double constraint_2 = poly[2] * 2 + poly[3] * 6 * _rho_powers[1] + poly[4] * 12 * _rho_powers[2];
        residuals[0] = poly[0] * _a + poly[2] * _a * _rho_powers[2] + poly[3] * _a * _rho_powers[3] + poly[4] * _a * _rho_powers[4] - t3[0] * _detected_corner[1] - _b;
        residuals[1] = poly[0] * _c + poly[2] * _c * _rho_powers[2] + poly[3] * _c * _rho_powers[3] + poly[4] * _c * _rho_powers[4] - t3[0] * _detected_corner[0] - _d;
        if ( constraint_1 < 0 || constraint_2 < 0 ) {
            residuals[0] = max ( -constraint_1, -constraint_2 ) * 1e100;
            residuals[1] = max ( -constraint_1, -constraint_2 ) * 1e100;
        }
        return true;
    }

    static ceres::CostFunction* Create ( const Vec2d& detected_corner, const Vec3d& board_corner, const Mat& transform ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToSolvePolyAndT3, ceres::CENTRAL, 2, POLY_SIZE, 1> (
                     new ErrorToSolvePolyAndT3 ( detected_corner, board_corner, transform )
                 ) );
    }
private:
    const Vec2d _detected_corner;
    const Vec3d _board_corner;
    const Mat _transform;
    vector<double> _rho_powers;
    double _a, _b, _c, _d;
};

struct ErrorToUpdateInversePoly {
public:
    ErrorToUpdateInversePoly ( const double* poly, const double rho )
        : _poly ( poly ), _rho ( rho ) {
        double f = Utils::EvaluatePolyEquation ( poly, POLY_SIZE, rho );
        _theta = atan2 ( f, rho );
    }

    bool operator() ( const double* inverse_poly, double* residuals ) const {
        double reprojected_rho = Utils::EvaluatePolyEquation ( inverse_poly, INV_POLY_SIZE, _theta );
        residuals[0] = _rho - reprojected_rho;
        return true;
    }

    static ceres::CostFunction* Create ( const double* poly, const double rho ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToUpdateInversePoly, ceres::CENTRAL, 1, INV_POLY_SIZE> (
                     new ErrorToUpdateInversePoly ( poly, rho )
                 ) );
    }
private:
    const double* _poly;
    double _rho, _theta;
};

struct ErrorToUpdatePoly {
public:
    ErrorToUpdatePoly ( const double* inverse_poly_parameters, const double* poly_parameters, const double theta )
        : _inverse_poly_parameters ( inverse_poly_parameters ), _poly_parameters ( poly_parameters ), _theta ( theta ) {
        _rho = Utils::EvaluatePolyEquation ( inverse_poly_parameters, INV_POLY_SIZE, theta );
    }

    bool operator() ( const double* poly_factors, double* residuals ) const {
        double new_poly_parameters[POLY_SIZE];
        copy ( _poly_parameters, _poly_parameters+POLY_SIZE, new_poly_parameters );
        for ( int i=0; i<POLY_SIZE; i++ ) {
            new_poly_parameters[i] *= poly_factors[i];
        }
        double f = Utils::EvaluatePolyEquation ( new_poly_parameters, POLY_SIZE, _rho );
        residuals[0] = _theta - atan2 ( f, _rho );
        return true;
    }

    static ceres::CostFunction* Create ( const double* inverse_poly_parameters, const double* poly_parameters, const double theta ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToUpdatePoly, ceres::CENTRAL, 1, POLY_SIZE> (
                     new ErrorToUpdatePoly ( inverse_poly_parameters, poly_parameters, theta )
                 ) );
    }
private:
    const double* _inverse_poly_parameters;
    const double* _poly_parameters;
    const double _theta;
    double _rho;
};

struct ErrorToOptimizeFully {
public:
    ErrorToOptimizeFully ( const Mat& detected_corners, const Mat& board_corners )
        : _detected_corners ( detected_corners ), _board_corners ( board_corners ) {}

    bool operator() ( const double* intrinsics, const double* rotation_vector, const double* translation_vector, double* residuals ) const {
        Mat reprojected_corners;
        Utils::ReprojectCornersInFrame ( intrinsics, rotation_vector, translation_vector, _board_corners, &reprojected_corners );
        Mat reprojection_error = reprojected_corners - _detected_corners;
        copy ( ( double* ) reprojection_error.data, ( double* ) reprojection_error.data + reprojected_corners.total(), residuals );
        return true;
    }

    static ceres::CostFunction* Create ( const Mat& detected_corners, const Mat& board_corners ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToOptimizeFully, ceres::CENTRAL, ceres::DYNAMIC, TOTAL_SIZE, 3, 3> (
                     new ErrorToOptimizeFully ( detected_corners, board_corners ), ceres::TAKE_OWNERSHIP, detected_corners.total()
                 ) );
    }

private:
    const Mat _detected_corners, _board_corners;
};

struct ErrorToOptimizeFully2 {
public:
    ErrorToOptimizeFully2 ( const Mat& detected_corners, const Mat& board_corners )
        : _detected_corners ( detected_corners ), _board_corners ( board_corners ) {}

    bool operator() ( const double* affine, const double* inverse_poly, const double* rotation_vector, const double* translation_vector, double* residuals ) const {
        double reconstructed_intrinsics[TOTAL_SIZE];
        copy ( affine, affine+AFFINE_SIZE, reconstructed_intrinsics );
        copy ( inverse_poly, inverse_poly+INV_POLY_SIZE, reconstructed_intrinsics+INV_POLY_START );
        Mat reprojected_corners;
        Utils::ReprojectCornersInFrame ( reconstructed_intrinsics, rotation_vector, translation_vector, _board_corners, &reprojected_corners );
        Mat reprojection_error = reprojected_corners - _detected_corners;
        copy ( ( double* ) reprojection_error.data, ( double* ) reprojection_error.data + reprojected_corners.total(), residuals );
        return true;
    }

    static ceres::CostFunction* Create ( const Mat& detected_corners, const Mat& board_corners ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToOptimizeFully2, ceres::CENTRAL, ceres::DYNAMIC, AFFINE_SIZE, INV_POLY_SIZE, 3, 3> (
                     new ErrorToOptimizeFully2 ( detected_corners, board_corners ), ceres::TAKE_OWNERSHIP, detected_corners.total()
                 ) );
    }

private:
    const Mat _detected_corners, _board_corners;
};

struct ErrorToOptimizeFully3 {
public:
    ErrorToOptimizeFully3 ( const Vec2d& detected_corner, const Vec3d& board_corner )
        : _detected_corner ( detected_corner ), _board_corner ( board_corner ) {}

    bool operator() ( const double* intrinsics, const double* rotation_vector, const double* translation_vector, double* residuals ) const {
        Vec2d reprojected_corner;
        Utils::ReprojectSingleCorner ( intrinsics, rotation_vector, translation_vector, _board_corner, &reprojected_corner );
        Vec2d reprojection_error = reprojected_corner - _detected_corner;
        residuals[0] = reprojection_error[0];
        residuals[1] = reprojection_error[1];
        return true;
    }

    static ceres::CostFunction* Create ( const Vec2d& detected_corner, const Vec3d& board_corner ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToOptimizeFully3, ceres::CENTRAL, 2, TOTAL_SIZE, 3, 3> (
                     new ErrorToOptimizeFully3 ( detected_corner, board_corner )
                 ) );
    }

private:
    const Vec2d _detected_corner;
    const Vec3d _board_corner;
};

struct ErrorToOptimizeSystemExtrinsics {
public:
    ErrorToOptimizeSystemExtrinsics ( const Vec2d& detected_corners, const Vec3d& board_corners, const double* intrinsics )
        :_detected_corner ( detected_corners ), _board_corner ( board_corners ), _intrinsics ( intrinsics ) {
    }

    bool operator() ( const double* camera_rotation_vector_data, const double* camera_translation_vector_data,
                      const double* frame_rotation_vector_data, const double* frame_translation_vector_data, double* residuals ) const {
        Mat camera_rotation_vector ( 3, 1, CV_64FC1, ( void* ) camera_rotation_vector_data );
        Mat camera_translation_vector ( 3, 1, CV_64FC1, ( void* ) camera_translation_vector_data );
        Mat frame_rotation_vector ( 3, 1, CV_64FC1, ( void* ) frame_rotation_vector_data );
        Mat frame_translation_vector ( 3, 1, CV_64FC1, ( void* ) frame_translation_vector_data );
        Mat camera_transform, frame_transform;
        Utils::GetTransformFromRAndTVectors ( camera_rotation_vector, camera_translation_vector, &camera_transform );
        Utils::GetTransformFromRAndTVectors ( frame_rotation_vector, frame_translation_vector, &frame_transform );
        camera_transform = Utils::GetTransform44From34 ( camera_transform );
        frame_transform = Utils::GetTransform44From34 ( frame_transform );
        Mat local_rotation_vector, local_translation_vector, local_transform = camera_transform * frame_transform;
        Utils::GetRAndTVectorsFromTransform ( local_transform, &local_rotation_vector, &local_translation_vector );

        Vec2d reprojected_corner;
        Utils::ReprojectSingleCorner ( _intrinsics, ( double* ) local_rotation_vector.data, ( double* ) local_translation_vector.data,
                                       _board_corner, &reprojected_corner );
        Vec2d reprojection_error = reprojected_corner - _detected_corner;
        residuals[0] = reprojection_error[0];
        residuals[1] = reprojection_error[1];
	return true;
    }

    static ceres::CostFunction* Create ( const Vec2d& detected_corner, const Vec3d& board_corner, const double* intrinsics ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToOptimizeSystemExtrinsics, ceres::CENTRAL, 2, 3, 3, 3, 3> (
                     new ErrorToOptimizeSystemExtrinsics ( detected_corner, board_corner, intrinsics )
                 ) );
    }

private:
    const Vec2d _detected_corner;
    const Vec3d _board_corner;
    const double* _intrinsics;
};

#endif // OPTIMIZATIONSTRUCTS_H

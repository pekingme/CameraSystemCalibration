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
        : detected_corner ( detected_corner ), board_corner ( board_corner ), transform ( transform ) {
        double rho = norm ( detected_corner );
        rho_powers.push_back ( 1 );
        for ( int i=1; i<=4; i++ ) {
            rho_powers.push_back ( rho_powers[i-1] * rho );
        }
        a = transform.at<double> ( 1,0 ) *board_corner[0]+transform.at<double> ( 1,1 ) *board_corner[1]+transform.at<double> ( 1,3 );
        c = transform.at<double> ( 0,0 ) *board_corner[0]+transform.at<double> ( 0,1 ) *board_corner[1]+transform.at<double> ( 0,3 );
        b = detected_corner[1] * ( transform.at<double> ( 2,0 ) *board_corner[0]+transform.at<double> ( 2,1 ) *board_corner[1] );
        d = detected_corner[0] * ( transform.at<double> ( 2,0 ) *board_corner[0]+transform.at<double> ( 2,1 ) *board_corner[1] );
    }

    bool operator() ( const double* poly, const double* t3, double* residuals ) const {
        double constraint_1 = poly[2] * 2 * rho_powers[1] + poly[3] * 3 * rho_powers[2] + poly[4] * 4 * rho_powers[3];
        double constraint_2 = poly[2] * 2 + poly[3] * 6 * rho_powers[1] + poly[4] * 12 * rho_powers[2];
        if ( constraint_1 < 0 || constraint_2 < 0 ) {
            residuals[0] = residuals[1] = 1e100;
        } else {
            residuals[0] = poly[0] * a + poly[2] * a * rho_powers[2] + poly[3] * a * rho_powers[3] + poly[4] * a * rho_powers[4] - t3[0] * detected_corner[1] - b;
            residuals[1] = poly[0] * c + poly[2] * c * rho_powers[2] + poly[3] * c * rho_powers[3] + poly[4] * c * rho_powers[4] - t3[0] * detected_corner[0] - d;
        }
        return true;
    }

    static ceres::CostFunction* Create ( const Vec2d& detected_corner, const Vec3d& board_corner, const Mat& transform ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToSolvePolyAndT3, ceres::CENTRAL, 2, POLY_SIZE, 1> (
                     new ErrorToSolvePolyAndT3 ( detected_corner, board_corner, transform )
                 ) );
    }
private:
    const Vec2d detected_corner;
    const Vec3d board_corner;
    const Mat transform;
    vector<double> rho_powers;
    double a, b, c, d;
};

struct ErrorToUpdateInversePoly {
public:
    ErrorToUpdateInversePoly ( const double* poly, const double rho )
        : poly ( poly ), rho ( rho ) {
        double f = Utils::EvaluatePolyEquation ( poly, POLY_SIZE, rho );
        theta = atan2 ( f, rho );
    }

    bool operator() ( const double* inverse_poly, double* residuals ) const {
        double reprojected_rho = Utils::EvaluatePolyEquation ( inverse_poly, INV_POLY_SIZE, theta );
        residuals[0] = rho - reprojected_rho;
        return true;
    }

    static ceres::CostFunction* Create ( const double* poly, const double rho ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToUpdateInversePoly, ceres::CENTRAL, 1, INV_POLY_SIZE> (
                     new ErrorToUpdateInversePoly ( poly, rho )
                 ) );
    }
private:
    const double* poly;
    double rho, theta;
};

struct ErrorToUpdatePoly {
public:
    ErrorToUpdatePoly ( const double* inverse_poly_parameters, const double* poly_parameters, const double theta )
        : inverse_poly_parameters ( inverse_poly_parameters ), poly_parameters ( poly_parameters ), theta ( theta ) {
        rho = Utils::EvaluatePolyEquation ( inverse_poly_parameters, INV_POLY_SIZE, theta );
    }

    bool operator() ( const double* poly_factors, double* residuals ) const {
        double new_poly_parameters[POLY_SIZE];
        copy ( poly_parameters, poly_parameters+POLY_SIZE, new_poly_parameters );
        for ( int i=0; i<POLY_SIZE; i++ ) {
            new_poly_parameters[i] *= poly_factors[i];
        }
        double f = Utils::EvaluatePolyEquation ( new_poly_parameters, POLY_SIZE, rho );
        residuals[0] = theta - atan2 ( f, rho );
        return true;
    }

    static ceres::CostFunction* Create ( const double* inverse_poly_parameters, const double* poly_parameters, const double theta ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToUpdatePoly, ceres::CENTRAL, 1, POLY_SIZE> (
                     new ErrorToUpdatePoly ( inverse_poly_parameters, poly_parameters, theta )
                 ) );
    }
private:
    const double* inverse_poly_parameters;
    const double* poly_parameters;
    const double theta;
    double rho;
};

struct ErrorToOptimizeFully {
public:
    ErrorToOptimizeFully ( const Mat& detected_corners, const Mat& board_corners )
        : detected_corners ( detected_corners ), board_corners ( board_corners ) {}

    bool operator() ( const double* intrinsics, const double* rotation_vector, const double* translation_vector, double* residuals ) const {
        Mat reprojected_corners;
        Utils::ReprojectCornersInFrame ( intrinsics, rotation_vector, translation_vector, board_corners, &reprojected_corners );
        Mat reprojection_error = reprojected_corners - detected_corners;
        copy ( ( double* ) reprojection_error.data, ( double* ) reprojection_error.data + reprojected_corners.total(), residuals );
        return true;
    }

    static ceres::CostFunction* Create ( const Mat& detected_corners, const Mat& board_corners ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToOptimizeFully, ceres::CENTRAL, ceres::DYNAMIC, TOTAL_SIZE, 3, 3> (
                     new ErrorToOptimizeFully ( detected_corners, board_corners ), ceres::TAKE_OWNERSHIP, detected_corners.total()
                 ) );
    }

private:
    const Mat detected_corners, board_corners;
};

struct ErrorToOptimizeFully2 {
public:
    ErrorToOptimizeFully2 ( const Mat& detected_corners, const Mat& board_corners )
        : detected_corners ( detected_corners ), board_corners ( board_corners ) {}

    bool operator() ( const double* affine, const double* inverse_poly, const double* rotation_vector, const double* translation_vector, double* residuals ) const {
        double reconstructed_intrinsics[TOTAL_SIZE];
        copy ( affine, affine+AFFINE_SIZE, reconstructed_intrinsics );
        copy ( inverse_poly, inverse_poly+INV_POLY_SIZE, reconstructed_intrinsics+INV_POLY_START );
        Mat reprojected_corners;
        Utils::ReprojectCornersInFrame ( reconstructed_intrinsics, rotation_vector, translation_vector, board_corners, &reprojected_corners );
        Mat reprojection_error = reprojected_corners - detected_corners;
        copy ( ( double* ) reprojection_error.data, ( double* ) reprojection_error.data + reprojected_corners.total(), residuals );
        return true;
    }

    static ceres::CostFunction* Create ( const Mat& detected_corners, const Mat& board_corners ) {
        return ( new ceres::NumericDiffCostFunction<ErrorToOptimizeFully2, ceres::CENTRAL, ceres::DYNAMIC, AFFINE_SIZE, INV_POLY_SIZE, 3, 3> (
                     new ErrorToOptimizeFully2 ( detected_corners, board_corners ), ceres::TAKE_OWNERSHIP, detected_corners.total()
                 ) );
    }

private:
    const Mat detected_corners, board_corners;
};

struct ErrorToOptimizeFully3 {
public:
    ErrorToOptimizeFully3 ( const Vec2d& detected_corner, const Vec3d& board_corner )
        : detected_corner ( detected_corner ), board_corner ( board_corner ) {}

    bool operator() ( const double* intrinsics, const double* rotation_vector, const double* translation_vector, double* residuals ) const {
        Vec2d reprojected_corner;
        Utils::ReprojectSingleCorner ( intrinsics, rotation_vector, translation_vector, board_corner, &reprojected_corner );
        Vec2d reprojection_error = reprojected_corner - detected_corner;
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
    const Vec2d detected_corner;
    const Vec3d board_corner;
};

#endif // OPTIMIZATIONSTRUCTS_H

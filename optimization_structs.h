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
        residuals[0] = poly[0] * a + poly[2] * a * rho_powers[2] + poly[3] * a * rho_powers[3] + poly[4] * a * rho_powers[4] - t3[0] * detected_corner[1] - b;
        residuals[1] = poly[0] * c + poly[2] * c * rho_powers[2] + poly[3] * c * rho_powers[3] + poly[4] * c * rho_powers[4] - t3[0] * detected_corner[0] - d;
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

struct ErrorToSolveInversePoly {
public:
    ErrorToSolveInversePoly ( const double* poly, const double rho )
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
        return ( new ceres::NumericDiffCostFunction<ErrorToSolveInversePoly, ceres::CENTRAL, 1, INV_POLY_SIZE> (
                     new ErrorToSolveInversePoly ( poly, rho )
                 ) );
    }
private:
    const double* poly;
    double rho, theta;
};

#endif // OPTIMIZATIONSTRUCTS_H

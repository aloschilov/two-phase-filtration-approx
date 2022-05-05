#include <iostream>

#import "newton_krylov.h"

int main() {
    double max_time = 3600*4;
    double max_x = float(10);

    const int nx = 10, nt = 10;
    const double hx = max_x / float(nx), ht = max_time / float(nt);

    const double m = 0.3, u = -0.0001;

    const double P_right = 1;
    const double P_bottom = 0;

    VecFunc residual = [&hx, &P_right, &m, &u, &P_bottom, &ht](Vec S) {
        Mat d1x = Mat::Zero(nx, nt);
        Mat d1t = Mat::Zero(nx, nt);
        Mat Sm = Eigen::Map<Mat>(S.data(), nx, nt);

        d1x.block(0,0,nx-1,nt) =
                (Sm.block(1,0,nx-1,nt) -
                Sm.block(0,0,nx-1,nt))/hx;

        d1x.block(nx-2,0,1,nt) =
                (P_right - Sm.block(nx-2,0,1,nt).array())/hx;


        d1t.block(0,0,nx,1) =
                (Sm.block(0,0,nx,1).array() - P_bottom)/ht;

        d1t.block(0,1,nx,nt-1) =
                (Sm.block(0,1,nx,nt-1) - Sm.block(0,0,nx,nt-1))/ht;
        Sm = (m * d1t.array() + (2* u * d1x.array()) / (1.0 + Sm.array()) - (2* u * Sm.array() * d1x.array()) / (1.0 + Sm.array()).pow(2)).matrix();

        return Eigen::Map<Vec>(Sm.data(), nx*nt);
    };

    Mat R = Mat::Zero(nx,nt);
    R.block(nx-1,0, 1,nt) = Mat::Ones(1,nt);
    Vec guess = Eigen::Map<Vec>(R.data(), nx*nt);
    double f_tol = 1e-6;
    double f_rtol = 1e-6;
    double x_tol = 1e-6;
    double x_rtol = 1e-6;
    Vec sol = nonlin_solve(residual, guess, f_tol, f_rtol, x_tol,x_rtol);

    Mat Sol = Eigen::Map<Mat>(sol.data(), nx, nt);

    std::cout << Sol << std::endl;
    return 0;
}

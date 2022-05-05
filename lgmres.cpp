#include <cmath>
#include <vector>

#include "lgmres.h"


//const double mEPS = 2.2204460492503131e-16;


Vec lgmres(VecFunc matvec, VecFunc psolve, Vecr b, Vec x,
           std::vector<Vec> &outer_v, const double tol, const int maxiter,
           const int inner_m, const unsigned int outer_k) {

    int n = b.size();

    double b_norm = b.norm();
    if (b_norm == 0.)
        b_norm = 1.;

    Vec r_outer(n);
    Vec vs0(n);
    Vec mvz(n);
    Vec z(n);
    Vec v_new(n);
    DecQR QR;

    for (int k_outer = 0; k_outer < maxiter; k_outer++) {

        r_outer = matvec(x) - b;

        double r_norm = r_outer.norm();
        if (r_norm <= tol * b_norm || r_norm <= tol)
            break;

        vs0 = -psolve(r_outer);
        double inner_res_0 = vs0.norm();
        vs0 /= inner_res_0;

        unsigned int ind = 1 + inner_m + outer_v.size();

        Mat vs = Mat(ind, n);
        Mat ws = Mat(ind - 1, n);
        Mat A = Mat(ind, ind);
        Mat Q = Mat::Zero(ind, ind);
        Mat R = Mat::Zero(ind, ind-1);
        Vec hcur = Vec(ind);
        vs.row(0) = vs0;
        Q(0, 0) = 1.;

        unsigned int j = 1;
        for (; j < ind; j++) {

            if (j < outer_v.size() + 1)
                z = outer_v[j - 1];
            else if (j == outer_v.size() + 1)
                z = vs0;
            else
                z = vs.row(j - 1);

            mvz = matvec(z);
            v_new = psolve(mvz);
            double v_new_norm = v_new.norm();

            hcur.head(j) = vs.topRows(j) * v_new;
            v_new -= hcur.transpose() * vs.topRows(j);
            hcur(j) = v_new.norm();

            v_new /= hcur(j);
            vs.row(j) = v_new;
            ws.row(j - 1) = z;

            Q(j, j) = 1.;

            A.topLeftCorner(j + 1, j - 1) = Q.topLeftCorner(j + 1, j + 1) *
                                            R.topLeftCorner(j + 1, j - 1);
            A.block(0, j - 1, j + 1, 1) = hcur.head(j + 1);
            QR.compute(A.topLeftCorner(j + 1, j));


            Q.topLeftCorner(j + 1, j + 1) = QR.householderQ();
            R.topLeftCorner(j + 1, j) = QR.matrixQR().triangularView<Eigen::Upper>();

            double inner_res = std::abs(Q(0, j)) * inner_res_0;

            if ((inner_res <= tol * inner_res_0) || (hcur(j) <= mEPS * v_new_norm))
                break;
        }
        if (j == ind)
            j -= 1;

        Vec y = R.topLeftCorner(j, j).householderQr().solve(
                Q.topLeftCorner(1, j).transpose());
        y *= inner_res_0;

        Vec dx = y(0) * ws.row(0);
        for (int i = 1; i < y.size(); i++)
            dx += y(i) * ws.row(i);

        double nx = dx.norm();
        if (nx > 0.)
            outer_v.push_back(dx / nx);

        while (outer_v.size() > outer_k)
            outer_v.erase(outer_v.begin());
        x += dx;
    }
    return x;
}

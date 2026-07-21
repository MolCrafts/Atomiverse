/// @file cpu/scf/solve.cpp
/// @brief CPU SCF solver: xtensor-blas reference implementation.

#include "atomiverse/cpu/scf/solve.h"

#include <cmath>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/views/xview.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace toy {

SCFResult solve(int n, int n_occ, double alpha, double tol, int max_iter) {
    // Build H0: deterministic diagonally-dominant symmetric matrix
    xt::xtensor<double, 2> H0 = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i) {
        H0(i, i) = double(n + 1 - i);
        for (int j = i + 1; j < n; ++j) {
            double val = double(1) / double(1 + (j - i));
            H0(i, j) = val;
            H0(j, i) = val;
        }
    }

    // Initialize density P = 0
    xt::xtensor<double, 2> P = xt::zeros<double>({n, n});
    xt::xtensor<double, 2> P_old = xt::zeros<double>({n, n});

    SCFResult result{};

    for (int iter = 0; iter < max_iter; ++iter) {
        // H = H0 + alpha * P
        xt::xtensor<double, 2> H = H0 + alpha * P;

        // Diagonalize H -> eigenvalues + eigenvectors
        auto [eig, C] = xt::linalg::eigh(H);

        // Save P_old, rebuild P = C_occ * C_occ^T
        P_old = P;
        auto C_occ = xt::view(C, xt::all(), xt::range(0, n_occ));
        P = xt::linalg::dot(C_occ, xt::transpose(C_occ));

        // Convergence: ||P - P_old||_F
        double norm = xt::norm_l2(P - P_old)();

        // Energy = trace(H * P) = sum of element-wise product
        double energy = xt::sum(H * P)();

        result.residual = norm;
        result.energy = energy;
        result.eigenvalues = eig;
        result.density = P;
        result.iterations = iter + 1;

        if (iter > 0 && norm < tol) {
            result.converged = true;
            break;
        }
    }

    return result;
}

} // namespace toy
} // namespace scf
} // namespace cpu
} // namespace ATV

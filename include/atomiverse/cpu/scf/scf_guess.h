/// @file cpu/scf/scf_guess.h
/// @brief SCF initial guesses: the optional density-matrix guess carrier
///        (`SCFInitialGuess`, rehomed from the since-deleted scf_result.h) and
///        broken-symmetry HOMO-LUMO mixing for locating spin-broken
///        solutions (e.g. stretched H2).
#pragma once

#include <cmath>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Optional density-matrix initial guess.
struct SCFInitialGuess {
    xt::xtensor<double, 2>
        P_alpha; ///< Alpha density guess [n x n] (dimensionless).
    xt::xtensor<double, 2>
        P_beta;             ///< Beta density guess [n x n] (dimensionless).
    bool has_alpha = false; ///< Whether P_alpha is set.
    bool has_beta = false;  ///< Whether P_beta is set.
};

/// @brief Rotate the HOMO and LUMO columns of C by `theta` to seed a broken-
///        symmetry guess (an asymmetric starting density).
/// @param C MO coefficient matrix [n x n], modified in place.
inline void apply_homo_lumo_mixing(RealMatrix& C, int n_occ, int n,
                                   double theta = 0.3490658503988659) {
    if (n_occ <= 0 || n_occ >= n)
        return;
    const double ca = std::cos(theta);
    const double sa = std::sin(theta);
    const int homo = n_occ - 1;
    const int lumo = n_occ;
    for (int r = 0; r < n; ++r) {
        const double ci = C(r, homo);
        const double cj = C(r, lumo);
        C(r, homo) = ca * ci + sa * cj;
        C(r, lumo) = -sa * ci + ca * cj;
    }
}

} // namespace scf
} // namespace cpu
} // namespace ATV

/// @file cpu/scf/spin_analysis.h
/// @brief Spin diagnostics for unrestricted wavefunctions.
#pragma once

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Spin contamination <S^2> - S(S+1) (zero for a pure spin state); a
///        nonzero value flags higher-spin admixture in the UHF/UKS solution.
inline double compute_spin_contamination(const RealMatrix& C_alpha,
                                         const RealMatrix& C_beta,
                                         const RealMatrix& S, int n_alpha,
                                         int n_beta) {
    if (n_alpha == 0 || n_beta == 0)
        return 0.0;
    auto Ca = xt::view(C_alpha, xt::all(), xt::range(0, n_alpha));
    auto Cb = xt::view(C_beta, xt::all(), xt::range(0, n_beta));
    RealMatrix SCb = xt::linalg::dot(S, Cb);
    RealMatrix O = xt::linalg::dot(xt::transpose(Ca), SCb);
    return static_cast<double>(n_beta) - xt::sum(O * O)();
}

} // namespace scf
} // namespace cpu
} // namespace ATV

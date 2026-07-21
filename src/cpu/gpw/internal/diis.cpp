/// @file cpu/gpw/internal/diis.cpp
/// @brief Single-channel Pulay DIIS for the restricted GPW KS matrix (see header).

#include "atomiverse/cpu/gpw/internal/diis.h"

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

RealMatrix gpw_diis_error(const RealMatrix& H, const RealMatrix& P,
                          const RealMatrix& S, const RealMatrix& X) {
    // Commutator residual H P S - S P H, transformed into the orthonormal basis.
    const RealMatrix commutator = xt::linalg::dot(H, xt::linalg::dot(P, S))
                                  - xt::linalg::dot(S, xt::linalg::dot(P, H));
    return xt::linalg::dot(xt::transpose(X), xt::linalg::dot(commutator, X));
}

RealMatrix GpwDiis::step(const RealMatrix& H, const RealMatrix& e) {
    h_hist_.push_back(H);
    e_hist_.push_back(e);
    while (static_cast<int>(h_hist_.size()) > max_vectors_) {
        h_hist_.pop_front();
        e_hist_.pop_front();
    }

    const int m = static_cast<int>(h_hist_.size());
    if (m < 2)
        return h_hist_.back(); // warm-up: not enough history to extrapolate

    // Pulay B-matrix: B_ij = Tr(e_i e_j), bordered by the unit-sum constraint.
    const int dim = m + 1;
    RealMatrix B = xt::zeros<double>({dim, dim});
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) {
            const double val = xt::sum(e_hist_[i] * e_hist_[j])();
            B(i, j) = val;
            B(j, i) = val;
        }
    for (int i = 0; i < m; ++i) {
        B(m, i) = -1.0;
        B(i, m) = -1.0;
    }
    B(m, m) = 0.0;

    RealMatrix rhs = xt::zeros<double>({dim, 1});
    rhs(m, 0) = -1.0;

    try {
        const RealMatrix c = xt::linalg::solve(B, rhs);
        const int n = static_cast<int>(H.shape(0));
        RealMatrix H_diis = xt::zeros<double>({n, n});
        for (int i = 0; i < m; ++i)
            H_diis += c(i, 0) * h_hist_[i];
        return H_diis;
    } catch (...) {
        // Singular subspace (e.g. linearly dependent errors): use the latest H.
        return h_hist_.back();
    }
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV

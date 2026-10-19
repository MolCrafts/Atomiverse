/// @file cpu/gpw/internal/diis.cpp
/// @brief Single-channel Pulay DIIS for the restricted GPW KS matrix (see header).

#include "atomiverse/cpu/gpw/internal/diis.h"
#include "atomiverse/cpu/linalg/pulay.h"

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

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
    RealMatrix B = xt::zeros<double>({m, m});
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) {
            const double val = xt::sum(e_hist_[i] * e_hist_[j])();
            B(i, j) = val;
            B(j, i) = val;
        }
    try {
        const auto c = linalg::pulay_coefficients(B);
        const int n = static_cast<int>(H.shape(0));
        RealMatrix H_diis = xt::zeros<double>({n, n});
        for (int i = 0; i < m; ++i)
            H_diis += c(i) * h_hist_[i];
        return H_diis;
    } catch (const std::runtime_error&) {
        // Singular subspace (e.g. linearly dependent errors): use the latest H.
        return h_hist_.back();
    }
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV

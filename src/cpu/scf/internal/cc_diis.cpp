/// @file cpu/scf/internal/cc_diis.cpp
/// @brief CcAmplitudeDiis — Pulay augmented-normal-equation solve for CC
///        amplitude extrapolation (see internal/cc_diis.h).

#include "atomiverse/cpu/scf/internal/cc_diis.h"

#include <cmath>
#include <cstddef>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

std::vector<double> CcAmplitudeDiis::step(const std::vector<double>& amp,
                                          const std::vector<double>& err) {
    amp_hist_.push_back(amp);
    err_hist_.push_back(err);
    while (static_cast<int>(amp_hist_.size()) > max_vectors_) {
        amp_hist_.pop_front();
        err_hist_.pop_front();
    }

    // Single stored vector → the extrapolant is the input verbatim (c_0 = 1),
    // matching DIISMixer's single-entry contract.
    if (amp_hist_.size() < 2)
        return amp_hist_.back();

    const std::size_t len = amp.size();

    // Drop the oldest history vector and retry whenever the Pulay solve returns
    // a non-finite extrapolant (near-parallel error vectors make B singular).
    while (amp_hist_.size() >= 2) {
        const int m = static_cast<int>(amp_hist_.size());
        const std::size_t dim = static_cast<std::size_t>(m) + 1;

        // Augmented Pulay system  [[B, -1], [-1^T, 0]] [c; λ] = [0; -1],
        // with B_ij = e_i · e_j over the flattened error vectors.
        xt::xtensor<double, 2> A = xt::zeros<double>({dim, dim});
        xt::xtensor<double, 2> rhs = xt::zeros<double>({dim, std::size_t{1}});
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j <= i; ++j) {
                double b = 0.0;
                const std::vector<double>& ei =
                    err_hist_[static_cast<std::size_t>(i)];
                const std::vector<double>& ej =
                    err_hist_[static_cast<std::size_t>(j)];
                for (std::size_t k = 0; k < ei.size(); ++k)
                    b += ei[k] * ej[k];
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = b;
                A(static_cast<std::size_t>(j), static_cast<std::size_t>(i)) = b;
            }
        }
        for (int i = 0; i < m; ++i) {
            A(static_cast<std::size_t>(m), static_cast<std::size_t>(i)) = -1.0;
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(m)) = -1.0;
        }
        A(static_cast<std::size_t>(m), static_cast<std::size_t>(m)) = 0.0;
        rhs(static_cast<std::size_t>(m), 0) = -1.0;

        bool ok = true;
        std::vector<double> out(len, 0.0);
        try {
            const xt::xtensor<double, 2> c = xt::linalg::solve(A, rhs);
            for (int i = 0; i < m; ++i) {
                const double ci = c(static_cast<std::size_t>(i), 0);
                if (!std::isfinite(ci)) {
                    ok = false;
                    break;
                }
                const std::vector<double>& Ti =
                    amp_hist_[static_cast<std::size_t>(i)];
                for (std::size_t k = 0; k < len; ++k)
                    out[k] += ci * Ti[k];
            }
        } catch (...) {
            ok = false;
        }

        if (ok) {
            for (double v : out) {
                if (!std::isfinite(v)) {
                    ok = false;
                    break;
                }
            }
        }
        if (ok)
            return out;

        // Singular / non-finite: drop the oldest pair and retry with a shorter
        // subspace (regularize-by-truncation).
        amp_hist_.pop_front();
        err_hist_.pop_front();
    }

    return amp_hist_.back();
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

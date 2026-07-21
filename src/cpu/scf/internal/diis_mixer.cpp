/// @file cpu/scf/internal/diis_mixer.cpp
/// @brief Combined DIIS accelerator implementation.

#include "atomiverse/cpu/scf/internal/diis_mixer.h"

#include <xtensor/reducers/xnorm.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

std::pair<RealMatrix, RealMatrix> DIISMixer::step(const RealMatrix& F_alpha,
                                                  const RealMatrix& e_alpha,
                                                  const RealMatrix& F_beta,
                                                  const RealMatrix& e_beta) {

    Fa_hist_.push_back(F_alpha);
    Fb_hist_.push_back(F_beta);
    ea_hist_.push_back(e_alpha);
    eb_hist_.push_back(e_beta);

    while (static_cast<int>(Fa_hist_.size()) > max_vectors_) {
        Fa_hist_.pop_front();
        Fb_hist_.pop_front();
        ea_hist_.pop_front();
        eb_hist_.pop_front();
    }

    int m = static_cast<int>(Fa_hist_.size());
    if (m < 2)
        return {Fa_hist_.back(), Fb_hist_.back()};

    int n = static_cast<int>(F_alpha.shape(0));

    // Build combined B matrix: B_ij = Tr(e^α_i · e^α_j) + Tr(e^β_i · e^β_j)
    int dim = m + 1;
    RealMatrix B = xt::zeros<double>({(int)dim, (int)dim});
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) {
            double val = xt::sum(ea_hist_[i] * ea_hist_[j])()
                         + xt::sum(eb_hist_[i] * eb_hist_[j])();
            B(i, j) = val;
            B(j, i) = val;
        }
    for (int i = 0; i < m; ++i) {
        B(m, i) = double(-1);
        B(i, m) = double(-1);
    }
    B(m, m) = double(0);

    RealMatrix rhs = xt::zeros<double>({(int)dim, (int)1});
    rhs(m, 0) = double(-1);

    try {
        auto c = xt::linalg::solve(B, rhs);
        RealMatrix Fa_diis = xt::zeros<double>({n, n});
        RealMatrix Fb_diis = xt::zeros<double>({n, n});
        for (int i = 0; i < m; ++i) {
            Fa_diis += double(c(i, 0)) * Fa_hist_[i];
            Fb_diis += double(c(i, 0)) * Fb_hist_[i];
        }
        return {Fa_diis, Fb_diis};
    } catch (...) {
        return {Fa_hist_.back(), Fb_hist_.back()};
    }
}

RealMatrix DIISMixer::step(const RealMatrix& F, const RealMatrix& e) {
    auto [Fa, Fb] = step(F, e, F, e);
    return Fa;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

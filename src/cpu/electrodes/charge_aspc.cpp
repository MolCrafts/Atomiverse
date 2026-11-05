/// @file cpu/electrodes/charge_aspc.cpp
/// @brief Kolafa ASPC predictor for electrode charges (CG warm start).
///
/// `order` K here counts K+1 history vectors (MetalWalls' convention, i.e.
/// Kolafa order K−1). The weights are `electronic::kolafa_aspc_weights(K+1)`
/// (Kolafa, J. Comput. Chem. 25, 335 (2004)):
///
///   Q_pred(n+1) = sum_{j=1}^{K+1} B_j Q(n+1-j)
///
/// `omega_[k]` stores B_{K+1-k}, i.e. index 0 = oldest of the window.
///
/// These are **not** the Lagrange (polynomial) extrapolation coefficients.
/// Kolafa's construction deliberately gives up exactness beyond degree 1
/// (sum_j B_j = 1, sum_j j B_j = 0, but sum_j j^2 B_j != 0) in exchange for
/// unconditional stability and time-reversibility of the predictor-corrector
/// pair; plain Lagrange extrapolation is exactly the scheme that paper shows
/// to be unstable in a self-consistent loop.
///
/// Oracle: MetalWalls (git c8ad175d, release 20.05) hard-codes the K=5 row of
/// this family in `src/electrode_charge.F90:572-577` —
/// B = {22/7, -55/14, 55/21, -22/21, 5/21, -1/42} — and applies it to the six
/// stored electrode-charge vectors (`QATOMS_NUM_STEPS = 7`,
/// `src/system.F90:107`) as the CG warm start.

#include "atomiverse/cpu/electrodes/charge_aspc.h"

#include <stdexcept>

#include "atomiverse/cpu/electronic/aspc_predictor.h"

namespace ATV {
namespace cpu {
namespace electrodes {

ChargeASPC::ChargeASPC(int order) : K_(order) {
    if (K_ < 0)
        throw std::invalid_argument("ChargeASPC: order must be ≥ 0");
    compute_coefficients_();
}

void ChargeASPC::compute_coefficients_() {
    // K+1 vectors; kolafa_aspc_weights is newest-first, omega_ oldest-first.
    const auto b = electronic::kolafa_aspc_weights(K_ + 1);
    omega_.assign(b.rbegin(), b.rend());
}

void ChargeASPC::record(const std::vector<double>& Q_scf) {
    if (Q_scf.empty())
        throw std::invalid_argument("ChargeASPC::record: empty charge vector");
    if (dim_ < 0)
        dim_ = static_cast<int>(Q_scf.size());
    else if (static_cast<int>(Q_scf.size()) != dim_)
        throw std::invalid_argument(
            "ChargeASPC::record: charge vector length changed");

    history_.push_back(Q_scf);
    while (static_cast<int>(history_.size()) > K_ + 2)
        history_.pop_front();
}

bool ChargeASPC::predict(std::vector<double>& Q_predicted) const {
    if (static_cast<int>(history_.size()) < K_ + 1)
        return false;

    const int offset = static_cast<int>(history_.size()) - (K_ + 1);
    const std::size_t n = history_.back().size();
    Q_predicted.assign(n, 0.0);
    for (int k = 0; k <= K_; ++k) {
        const double w = omega_[static_cast<std::size_t>(k)];
        const auto& hk = history_[static_cast<std::size_t>(offset + k)];
        for (std::size_t i = 0; i < n; ++i)
            Q_predicted[i] += w * hk[i];
    }
    return true;
}

void ChargeASPC::reset() {
    history_.clear();
    dim_ = -1;
}

double ChargeASPC::coefficient(int k) const {
    if (k < 0 || k > K_)
        throw std::out_of_range("ChargeASPC::coefficient");
    return omega_[static_cast<std::size_t>(k)];
}

} // namespace electrodes
} // namespace cpu
} // namespace ATV

/// @file cpu/electrodes/charge_aspc.cpp
/// @brief Kolafa ASPC (Always-Stable Predictor-Corrector) for electrode charges.
///
/// Predictor coefficients for order K (Kolafa, J. Comput. Chem. 25, 335 (2004),
/// Eq. 10), applied to the K+1 most recent history vectors:
///
///   B_j = (-1)^(j+1) * j * C(2K+2, K+1-j) / C(2K, K),   j = 1 … K+1
///   Q_pred(n+1) = sum_j B_j Q(n+1-j)
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

#include <limits>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace electrodes {

namespace {

/// @brief Exact binomial coefficient \(\binom{n}{k}\).
/// @param n Upper index.
/// @param k Lower index.
/// @return \(\binom{n}{k}\), or 0 when \(k\) is out of \([0,n]\).
/// @throws std::overflow_error if the value exceeds 64-bit range.
unsigned long long binomial(int n, int k) {
    if (k < 0 || k > n)
        return 0ULL;
    if (k > n - k)
        k = n - k;
    unsigned long long r = 1ULL;
    for (int i = 1; i <= k; ++i) {
        const auto num = static_cast<unsigned long long>(n - k + i);
        if (r > std::numeric_limits<unsigned long long>::max() / num)
            throw std::overflow_error("ChargeASPC: order too large");
        r = r * num / static_cast<unsigned long long>(i); // exact at each step
    }
    return r;
}

} // namespace

ChargeASPC::ChargeASPC(int order) : K_(order) {
    if (K_ < 0)
        throw std::invalid_argument("ChargeASPC: order must be ≥ 0");
    compute_coefficients_();
}

void ChargeASPC::compute_coefficients_() {
    omega_.resize(static_cast<std::size_t>(K_ + 1));
    // Single division per coefficient keeps the value bit-identical to the
    // exact rational (e.g. 792/252 == 22/7 as doubles) so the K=5 row matches
    // MetalWalls' hard-coded constants exactly.
    const double denom = static_cast<double>(binomial(2 * K_, K_));
    for (int j = 1; j <= K_ + 1; ++j) {
        const double num =
            static_cast<double>(j)
            * static_cast<double>(binomial(2 * K_ + 2, K_ + 1 - j));
        const double sign = (j % 2 == 1) ? 1.0 : -1.0;
        // omega_ index 0 = oldest of the K+1 window = B_{K+1}.
        omega_[static_cast<std::size_t>(K_ + 1 - j)] = sign * num / denom;
    }
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

void ChargeASPC::correct(const std::vector<double>& Q_scf,
                         const std::vector<double>& Q_predicted,
                         std::vector<double>& Q_corrected) const {
    if (Q_scf.size() != Q_predicted.size())
        throw std::invalid_argument("ChargeASPC::correct: size mismatch");
    const double w_scf =
        static_cast<double>(K_ + 2) / static_cast<double>(2 * K_ + 3);
    const double w_pred =
        static_cast<double>(K_ + 1) / static_cast<double>(2 * K_ + 3);
    Q_corrected.resize(Q_scf.size());
    for (std::size_t i = 0; i < Q_scf.size(); ++i)
        Q_corrected[i] = w_scf * Q_scf[i] + w_pred * Q_predicted[i];
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

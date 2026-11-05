/// @file src/cpu/electronic/aspc_predictor.cpp
/// @brief ASPC predictor-corrector implementation.
///
/// Order K predicts from the K+2 newest snapshots with Kolafa's weights
/// (`kolafa_aspc_weights(K+2)`) and blends the corrector result with
/// (K+2)/(2K+3).
///
/// Reference: Kolafa, J. Comput. Chem. 25, 335 (2004); CP2K
///            qs_wf_history_methods.F (wfi_aspc_nr)

#include "atomiverse/cpu/electronic/aspc_predictor.h"

#include <limits>

#include <stdexcept>

namespace ATV {
namespace cpu {
namespace electronic {

namespace {

/// @brief Exact binomial coefficient C(n, k); 0 when k is outside [0, n].
unsigned long long binomial(int n, int k) {
    if (k < 0 || k > n)
        return 0ULL;
    if (k > n - k)
        k = n - k;
    unsigned long long r = 1ULL;
    for (int i = 1; i <= k; ++i) {
        const auto num = static_cast<unsigned long long>(n - k + i);
        if (r > std::numeric_limits<unsigned long long>::max() / num)
            throw std::overflow_error("kolafa_aspc_weights: order too large");
        r = r * num / static_cast<unsigned long long>(i); // exact at each step
    }
    return r;
}

} // namespace

std::vector<double> kolafa_aspc_weights(int n_vectors) {
    if (n_vectors < 1)
        throw std::invalid_argument(
            "kolafa_aspc_weights: n_vectors must be >= 1");
    const int n = n_vectors;
    // One division per weight keeps exact rationals bit-identical
    // (e.g. 792/252 == 22/7 for the six-vector MetalWalls row).
    const double denom = static_cast<double>(binomial(2 * n - 2, n - 1));
    std::vector<double> w(static_cast<std::size_t>(n));
    for (int j = 1; j <= n; ++j) {
        const double sign = (j % 2 == 1) ? 1.0 : -1.0;
        w[static_cast<std::size_t>(j - 1)] =
            sign * static_cast<double>(j)
            * static_cast<double>(binomial(2 * n, n - j)) / denom;
    }
    return w;
}

ASPCPredictor::ASPCPredictor(int order) : K_(order) {
    if (K_ < 0)
        throw std::invalid_argument("ASPC order must be non-negative");
    compute_coefficients();
}

void ASPCPredictor::compute_coefficients() {
    omega_ = kolafa_aspc_weights(K_ + 2); // B_1 (newest) … B_{K+2}
}

void ASPCPredictor::record(const xt::xtensor<double, 2>& P_scf) {
    history_.push_back(P_scf);
    while (static_cast<int>(history_.size()) > K_ + 2)
        history_.pop_front();
}

bool ASPCPredictor::predict(xt::xtensor<double, 2>& P_predicted) const {
    const int n = K_ + 2;
    if (static_cast<int>(history_.size()) < n)
        return false;
    const int newest = static_cast<int>(history_.size()) - 1;
    P_predicted = xt::zeros_like(history_.back());
    for (int j = 1; j <= n; ++j)
        P_predicted += omega_[static_cast<std::size_t>(j - 1)]
                       * history_[static_cast<std::size_t>(newest - (j - 1))];
    return true;
}

void ASPCPredictor::correct(const xt::xtensor<double, 2>& P_scf,
                            const xt::xtensor<double, 2>& P_predicted,
                            xt::xtensor<double, 2>& P_corrected) const {
    // Kolafa corrector: weighted blend for time-reversibility
    // P_corr = (K+2)/(2K+3) * P_scf + (K+1)/(2K+3) * P_pred
    double w_scf =
        static_cast<double>(K_ + 2) / static_cast<double>(2 * K_ + 3);
    double w_pred =
        static_cast<double>(K_ + 1) / static_cast<double>(2 * K_ + 3);
    P_corrected = w_scf * P_scf + w_pred * P_predicted;
}

void ASPCPredictor::reset() { history_.clear(); }

// ── complex (multi-k P_k) path ────────────────────────────────────────────

ASPCPredictorCplx::ASPCPredictorCplx(int order) : K_(order) {
    if (K_ < 0)
        throw std::invalid_argument("ASPC order must be non-negative");
    compute_coefficients();
}

void ASPCPredictorCplx::compute_coefficients() {
    omega_ = kolafa_aspc_weights(K_ + 2); // B_1 (newest) … B_{K+2}
}

void ASPCPredictorCplx::record(const linalg::CplxMatrix& P_scf) {
    history_.push_back(P_scf);
    while (static_cast<int>(history_.size()) > K_ + 2)
        history_.pop_front();
}

bool ASPCPredictorCplx::predict(linalg::CplxMatrix& P_predicted) const {
    const int n = K_ + 2;
    if (static_cast<int>(history_.size()) < n)
        return false;
    const int newest = static_cast<int>(history_.size()) - 1;
    using Cplx = std::complex<double>;
    P_predicted = xt::zeros_like(history_.back());
    for (int j = 1; j <= n; ++j)
        P_predicted += Cplx(omega_[static_cast<std::size_t>(j - 1)])
                       * history_[static_cast<std::size_t>(newest - (j - 1))];
    return true;
}

void ASPCPredictorCplx::correct(const linalg::CplxMatrix& P_scf,
                                const linalg::CplxMatrix& P_predicted,
                                linalg::CplxMatrix& P_corrected) const {
    using Cplx = std::complex<double>;
    const double w_scf =
        static_cast<double>(K_ + 2) / static_cast<double>(2 * K_ + 3);
    const double w_pred =
        static_cast<double>(K_ + 1) / static_cast<double>(2 * K_ + 3);
    P_corrected = Cplx(w_scf) * P_scf + Cplx(w_pred) * P_predicted;
}

void ASPCPredictorCplx::reset() { history_.clear(); }

} // namespace electronic
} // namespace cpu
} // namespace ATV

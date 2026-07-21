/// @file cpu/scf/aspc_predictor.cpp
/// @brief ASPC predictor-corrector implementation.
///
/// Lagrange extrapolation coefficients for order K:
///   omega_k = product_{j=0..K, j!=k} (K+1-j)/(k-j)
/// evaluated at the prediction point (K+1 steps ahead of the oldest point).
///
/// Reference: Kolafa, J. Comput. Chem. 25, 335 (2004), Eq. 7-11

#include "atomiverse/cpu/scf/aspc_predictor.h"

#include <stdexcept>

namespace ATV {
namespace cpu {
namespace scf {

ASPCPredictor::ASPCPredictor(int order) : K_(order) {
    if (K_ < 0)
        throw std::invalid_argument("ASPC order must be non-negative");
    compute_coefficients();
}

void ASPCPredictor::compute_coefficients() {
    // Lagrange extrapolation coefficients for predicting at point (K+1)
    // given history at points 0, 1, ..., K (newest = K).
    //
    // omega_k = product_{j=0..K, j!=k} ((K+1) - j) / (k - j)
    omega_.resize(K_ + 1);
    for (int k = 0; k <= K_; ++k) {
        double num = 1, den = 1;
        for (int j = 0; j <= K_; ++j) {
            if (j == k)
                continue;
            num *= (K_ + 1) - j;
            den *= k - j;
        }
        omega_[k] = num / den;
    }
}

void ASPCPredictor::record(const xt::xtensor<double, 2>& P_scf) {
    history_.push_back(P_scf);
    // Keep at most K+2 entries (K+1 for prediction + 1 spare)
    while (static_cast<int>(history_.size()) > K_ + 2)
        history_.pop_front();
}

bool ASPCPredictor::predict(xt::xtensor<double, 2>& P_predicted) const {
    // Need at least K+1 history points
    if (static_cast<int>(history_.size()) < K_ + 1)
        return false;

    int offset = static_cast<int>(history_.size()) - (K_ + 1);
    P_predicted = xt::zeros_like(history_.back());

    for (int k = 0; k <= K_; ++k)
        P_predicted += omega_[k] * history_[offset + k];

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

} // namespace scf
} // namespace cpu
} // namespace ATV

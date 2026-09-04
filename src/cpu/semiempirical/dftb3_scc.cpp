/// @file src/cpu/semiempirical/dftb3_scc.cpp
/// DFTB3 second-/third-order SCC energy + atomic potential (private workspace).
#include "atomiverse/cpu/semiempirical/internal/dftb3_scc.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/cpu/semiempirical/internal/dftb_gamma.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

Dftb3Scc::Dftb3Scc(std::vector<double> hubbard_u,
                   std::vector<double> hubbard_derivative,
                   std::vector<double> distance)
    : u_(std::move(hubbard_u)), ud_(std::move(hubbard_derivative)),
      r_(std::move(distance)) {
    if (u_.empty())
        throw std::invalid_argument("Dftb3Scc: empty Hubbard U vector");
    if (ud_.size() != u_.size())
        throw std::invalid_argument(
            "Dftb3Scc: Hubbard derivative size " + std::to_string(ud_.size())
            + " != Hubbard U size " + std::to_string(u_.size()));
    if (r_.size() != u_.size() * u_.size())
        throw std::invalid_argument("Dftb3Scc: distance matrix size "
                                    + std::to_string(r_.size()) + " != n^2 = "
                                    + std::to_string(u_.size() * u_.size()));
    for (double ud : ud_) {
        if (ud == 0.0 || !std::isfinite(ud))
            throw std::invalid_argument(
                "Dftb3Scc: Hubbard derivative must be filled and all nonzero");
    }
    n_ = static_cast<int>(u_.size());
}

void Dftb3Scc::require_size(const std::vector<double>& dq) const {
    if (dq.size() != u_.size())
        throw std::invalid_argument(
            "Dftb3Scc: charge fluctuation size " + std::to_string(dq.size())
            + " != atom count " + std::to_string(u_.size()));
}

double Dftb3Scc::gamma(int a, int b) const {
    const std::size_t ia = static_cast<std::size_t>(a);
    const std::size_t ib = static_cast<std::size_t>(b);
    if (a == b)
        return u_[ia];
    // DFTB+ Elstner form (same kernel as DFTB2) — see internal/dftb_gamma.h.
    return dftb_gamma::gamma_eV(r_[ia * u_.size() + ib], u_[ia], u_[ib]);
}

double Dftb3Scc::gamma_charge_derivative(int a, int b) const {
    const std::size_t ia = static_cast<std::size_t>(a);
    const std::size_t ib = static_cast<std::size_t>(b);
    if (a == b)
        return ud_[ia]; // γ_AA = U_A ⇒ ∂γ_AA/∂q_A = dU_A/dq_A
    return dftb_gamma::gamma_charge_eV(r_[ia * u_.size() + ib], u_[ia], u_[ib],
                                       ud_[ia]);
}

double Dftb3Scc::gamma_distance_derivative(int a, int b) const {
    if (a == b)
        return 0.0; // γ_AA = U_A carries no geometry
    const std::size_t ia = static_cast<std::size_t>(a);
    const std::size_t ib = static_cast<std::size_t>(b);
    return dftb_gamma::dgamma_dr_eV(r_[ia * u_.size() + ib], u_[ia], u_[ib]);
}

double Dftb3Scc::gamma_charge_distance_derivative(int a, int b) const {
    if (a == b)
        return 0.0; // Γ_AA = U^d_A carries no geometry
    const std::size_t ia = static_cast<std::size_t>(a);
    const std::size_t ib = static_cast<std::size_t>(b);
    return dftb_gamma::gamma_charge_dr_eV(r_[ia * u_.size() + ib], u_[ia],
                                          u_[ib], ud_[ia]);
}

double
Dftb3Scc::pair_energy_distance_derivative(int a, int b,
                                          const std::vector<double>& dq) const {
    require_size(dq);
    if (a == b)
        return 0.0;
    const double dqa = dq[static_cast<std::size_t>(a)];
    const double dqb = dq[static_cast<std::size_t>(b)];
    // E²: the ordered double sum contributes ½·2 = 1 × γ_AB Δq_A Δq_B.
    // E³: the two orderings are NOT the same term — Γ_AB ≠ Γ_BA.
    return dqa * dqb * gamma_distance_derivative(a, b)
           + (1.0 / 3.0)
                 * (dqa * dqa * dqb * gamma_charge_distance_derivative(a, b)
                    + dqb * dqb * dqa * gamma_charge_distance_derivative(b, a));
}

double Dftb3Scc::second_order_energy(const std::vector<double>& dq) const {
    require_size(dq);
    double e2 = 0;
    for (int a = 0; a < n_; ++a)
        for (int b = 0; b < n_; ++b)
            e2 += 0.5 * gamma(a, b) * dq[static_cast<std::size_t>(a)]
                  * dq[static_cast<std::size_t>(b)];
    return e2;
}

double Dftb3Scc::third_order_energy(const std::vector<double>& dq) const {
    require_size(dq);
    double e3 = 0;
    for (int a = 0; a < n_; ++a) {
        const double dqa = dq[static_cast<std::size_t>(a)];
        for (int b = 0; b < n_; ++b)
            e3 += (1.0 / 3.0) * gamma_charge_derivative(a, b) * dqa * dqa
                  * dq[static_cast<std::size_t>(b)];
    }
    return e3;
}

std::vector<double>
Dftb3Scc::second_order_potential(const std::vector<double>& dq) const {
    require_size(dq);
    std::vector<double> v(static_cast<std::size_t>(n_), 0.0);
    for (int a = 0; a < n_; ++a) {
        double s = 0;
        for (int b = 0; b < n_; ++b)
            s += gamma(a, b) * dq[static_cast<std::size_t>(b)];
        v[static_cast<std::size_t>(a)] = s;
    }
    return v;
}

std::vector<double>
Dftb3Scc::third_order_potential(const std::vector<double>& dq) const {
    require_size(dq);
    std::vector<double> v(static_cast<std::size_t>(n_), 0.0);
    // V³_C = ⅔ Δq_C Σ_B Γ_CB Δq_B + ⅓ Σ_B Δq_B² Γ_BC.
    for (int c = 0; c < n_; ++c) {
        const double dqc = dq[static_cast<std::size_t>(c)];
        double s = 0;
        for (int b = 0; b < n_; ++b) {
            const double dqb = dq[static_cast<std::size_t>(b)];
            s += (2.0 / 3.0) * dqc * gamma_charge_derivative(c, b) * dqb;
            s += (1.0 / 3.0) * dqb * dqb * gamma_charge_derivative(b, c);
        }
        v[static_cast<std::size_t>(c)] = s;
    }
    return v;
}

std::vector<double> Dftb3Scc::potential(const std::vector<double>& dq) const {
    std::vector<double> v = second_order_potential(dq);
    const std::vector<double> v3 = third_order_potential(dq);
    for (std::size_t a = 0; a < v.size(); ++a)
        v[a] += v3[a];
    return v;
}

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV

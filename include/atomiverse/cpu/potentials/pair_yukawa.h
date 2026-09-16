/// @file cpu/potentials/pair_yukawa.h
/// @brief Screened Coulomb (Yukawa) pair: U = A exp(−κr)/r (CPU).
#pragma once

#include <string>

#include "atomiverse/cpu/potentials/pair.h"

namespace ATV {
namespace cpu {

/// @brief Yukawa pair potential with optional energy shift at cutoff.
///
/// \(U(r)=A e^{-\kappa r}/r - U_{\mathrm{shift}}\), \(r < r_c\).
/// \(A>0\) is repulsive. Units: energy [eV], length [Å], \(\kappa\) [Å⁻¹].
class PairYukawa : public Pair {
  public:
    /// @param A Amplitude [eV·Å] (prefactor of exp(−κr)/r).
    /// @param kappa Screening wavenumber [Å⁻¹] (≥ 0).
    /// @param cutoff Pair cutoff [Å] (> 0).
    PairYukawa(double A, double kappa, double cutoff);

    std::string name() const override { return "yukawa"; }

    const NeighborConsumer* as_neighbor_consumer() const final {
        return nullptr;
    }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double A_ = 0;
    double kappa_ = 0;
    double cutoff_sq_ = 0;
    double v_shift_ = 0;
};

} // namespace cpu
} // namespace ATV

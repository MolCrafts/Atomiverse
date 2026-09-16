/// @file cpu/potentials/pair_buck.h
/// @brief Buckingham (Born–Mayer + dispersion) pair: U = A exp(−r/ρ) − C/r⁶ (CPU).
///
/// Matches LAMMPS `pair_style buck` with energy shift at cutoff
/// (`pair_modify shift yes`): \(U(r_c)=0\). Units: energy [eV], length [Å].
#pragma once

#include <string>

#include "atomiverse/cpu/potentials/pair.h"

namespace ATV {
namespace cpu {

/// @brief Buckingham pair potential (single-type, shifted at cutoff).
///
/// \[
/// U(r)=A\,e^{-r/\rho}-\frac{C}{r^{6}}-U_{\mathrm{shift}},\quad r < r_c
/// \]
/// with \(U_{\mathrm{shift}}=A e^{-r_c/\rho}-C/r_c^{6}\).
///
/// Force: \(F=-\nabla U\) with
/// \(\mathrm{d}U/\mathrm{d}r=-(A/\rho)\,e^{-r/\rho}+6C/r^{7}\) (unshifted;
/// shift is constant).
///
/// O(N²) all-pairs walk (NeighborConsumer withdrawn) — same foothold shape as
/// `PairYukawa` / `PairWCA`. Production multi-type + NL is deferred to Fumi–Tosi.
class PairBuck : public Pair {
  public:
    /// @param A      Born–Mayer prefactor [eV] (must be ≥ 0).
    /// @param rho    Softness length [Å] (must be > 0).
    /// @param C      Dispersion coefficient [eV·Å⁶] (must be ≥ 0).
    /// @param cutoff Pair cutoff [Å] (must be > 0).
    PairBuck(double A, double rho, double C, double cutoff);

    std::string name() const override { return "buck"; }

    /// @brief O(N²) path: withdraw neighbor-list gate.
    const NeighborConsumer* as_neighbor_consumer() const final {
        return nullptr;
    }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double A_ = 0;
    double rho_ = 0;
    double C_ = 0;
    double cutoff_sq_ = 0;
    double v_shift_ = 0; ///< U(r_c) so shifted potential is zero at cutoff
};

} // namespace cpu
} // namespace ATV

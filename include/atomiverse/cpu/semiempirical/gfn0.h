/// @file cpu/semiempirical/gfn0.h
/// @brief GFN0-xTB energy, analytic nuclear forces, and FD Hessian (H/C/N/O).
#pragma once

#include <string>

#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief GFN0-xTB non-SCC energy and analytic nuclear forces for H/C/N/O.
///
/// Production GFN0 (Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv 2019,
/// DOI 10.26434/chemrxiv.8326202.v1): erf coordination number, electronegativity
/// equilibration charges, one extended-Hückel diagonalization of a
/// CN- and charge-shifted Hamiltonian, classical repulsion, two-body D4(BJ)
/// and the polar-bond short-range correction. Not SCC-DFTB; not GFN1/GFN2.
/// Static tables: `basis_data/gfn0_tables.h` (H/C/N/O). Unsupported Z throws.
///
/// Oracle: **xtb** `--gfn 0` (never DFTB+ / OpenMOPAC). Units on the public
/// surface: energy [eV], positions [Å], forces [eV/Å].
///
/// @code
///   cpu::semiempirical::GFN0 gfn0;
///   gfn0.with_forces = true;
///   eval.on(evaluate::Phase::evaluate).add(std::move(gfn0));
///   // FD Hessian of the analytic G: cpu::vib::Hessian + on(vib::Phase::force).
/// @endcode
class GFN0 : public cpu::Semiempirical {
  public:
    /// @brief Molecular charge [e] (EEQ constraint).
    int charge = 0;
    /// @brief When true, write analytic forces F=−∇E [eV/Å] on launch.
    bool with_forces = false;

    GFN0() = default;

    std::string name() const override { return "gfn0"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    /// @brief Last electronic+classical energy [eV].
    double last_energy() const { return energy_; }

  private:
    double energy_ = 0;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV

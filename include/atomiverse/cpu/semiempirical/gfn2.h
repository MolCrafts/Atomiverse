/// @file cpu/semiempirical/gfn2.h
/// @brief GFN2-xTB energy, analytic nuclear forces, and FD Hessian (H/C/N/O).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief GFN2-xTB energy and analytic nuclear forces for H/C/N/O
///        (Bannwarth, Ehlert, Grimme, JCTC 2019).
///
/// Molecular closed-shell energy: STO-*n*G valence basis, CN'-shifted H⁰
/// (double-exponential CN), shell-resolved isotropic SCC, on-site
/// third-order, anisotropic electrostatics (charge–dipole, dipole–dipole,
/// charge–quadrupole) + dipole/quadrupole AXC on cumulative atomic
/// multipoles of every atom, screened repulsion and self-consistent D4
/// (C6 scaled by the SCC charges). AES and D4 enter the Fock as
/// \(\partial E/\partial P\), so the converged energy is stationary in
/// the density and the analytic gradient is its exact derivative
/// (Hellmann–Feynman + Pulay with frozen-\(P\) multipole chains).
/// Parameters are static in-tree tables — there is no `libxtb` kernel.
/// Oracle: **xtb** 6.7.1. Units: energy [eV], positions [Å], forces [eV/Å].
///
/// @code
///   cpu::semiempirical::GFN2 gfn2;
///   gfn2.with_forces = true;
///   eval.on(evaluate::Phase::evaluate).add(std::move(gfn2));
///   // FD Hessian of the analytic G: cpu::vib::Hessian + on(vib::Phase::force).
/// @endcode
class GFN2 : public cpu::Semiempirical, public cpu::QmEngine {
  public:
    /// @brief SCC tolerance on max |Δ| of the shell populations [e] and the
    ///        atomic dipoles/quadrupoles [e·Bohr, e·Bohr²] between cycles.
    double scc_tol = 1e-8;
    /// @brief Maximum SCC iterations. Exhausting it throws.
    int max_scc = 50;
    /// @brief Anderson mixing β ∈ (0, 1] (populations + multipoles).
    double mix_beta = 0.4;
    /// @brief Net molecular charge [e].
    const int charge;
    /// @brief When true, write analytic forces \(F=-\nabla E\) [eV/Å] on launch.
    bool with_forces = false;

    /// @brief GFN2-xTB.
    /// @param charge Net molecular charge [e].
    explicit GFN2(int charge = 0) : charge(charge) {}

    std::string name() const override { return "gfn2"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    std::string qm_engine_name() const override { return name(); }
    void prepare(const BuildContext& qm_ctx) override { build(qm_ctx); }
    void evaluate(const LaunchContext& qm_ctx) override { launch(qm_ctx); }

    /// @brief Last total energy [eV].
    double last_energy() const { return energy_; }
    /// @brief Last SCC residual (populations and multipoles).
    double last_scc_residual() const { return scc_residual_; }
    /// @brief SCC iterations of the last solve.
    int last_scc_iterations() const { return scc_iterations_; }
    /// @brief Last Mulliken Δq_A [e] (n0 − n).
    const std::vector<double>& last_mulliken_dq() const { return last_dq_; }

  private:
    double energy_ = 0;
    double scc_residual_ = 0;
    int scc_iterations_ = 0;
    std::vector<double> last_dq_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV

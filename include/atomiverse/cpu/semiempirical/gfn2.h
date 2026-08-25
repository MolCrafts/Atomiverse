/// @file cpu/semiempirical/gfn2.h
/// @brief GFN2-xTB energy (static in-tree tables, CPU).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief GFN2-xTB energy for H/C/N/O (Bannwarth, Ehlert, Grimme, JCTC 2019).
///
/// Molecular closed-shell energy: STO-*n*G valence basis, CN-shifted H⁰,
/// shell-resolved isotropic SCC, on-site third-order, anisotropic
/// electrostatics + AXC, screened repulsion, Grimme D4 (GFN2 damping).
/// Parameters are static in-tree tables — there is no `libxtb` kernel.
/// Oracle: **xtb** 6.7.1 (`reg_gfn2-xtb-water`). Units: energy [eV],
/// positions [Å]. Forces are not this slug (`gfn2-xtb-forces`).
///
/// @code
///   cpu::semiempirical::GFN2 gfn2;
///   gfn2.with_forces = false;
///   eval.on(evaluate::Phase::evaluate).add(std::move(gfn2));
/// @endcode
class GFN2 : public cpu::Semiempirical, public cpu::QmEngine {
  public:
    /// @brief SCC residual tolerance on max |Δq_shell| [e].
    double scc_tol = 1e-8;
    /// @brief Maximum SCC iterations. Exhausting it throws.
    int max_scc = 50;
    /// @brief Anderson charge-mixing β ∈ (0, 1].
    double mix_beta = 0.4;
    /// @brief Net molecular charge [e].
    int charge = 0;
    /// @brief When true, throw — analytic / FD forces are `gfn2-xtb-forces`.
    bool with_forces = false;

    GFN2() = default;

    std::string name() const override { return "gfn2"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    std::string qm_engine_name() const override { return name(); }
    void prepare(const BuildContext& qm_ctx) override { build(qm_ctx); }
    void evaluate(const LaunchContext& qm_ctx) override { launch(qm_ctx); }

    /// @brief Last total energy [eV].
    double last_energy() const { return energy_; }
    /// @brief Last SCC residual max|Δq| [e].
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

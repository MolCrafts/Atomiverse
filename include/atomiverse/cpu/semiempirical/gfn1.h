/// @file cpu/semiempirical/gfn1.h
/// @brief GFN1-xTB energy (organic H/C/N/O, static in-tree tables, CPU).
#pragma once

#include <string>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief GFN1-xTB total energy for H/C/N/O.
///
/// First-generation Grimme extended tight-binding (Grimme, Bannwarth,
/// Shushkov, *J. Chem. Theory Comput.* **13**, 1989 (2017), DOI
/// 10.1021/acs.jctc.7b00118): EHT Hamiltonian on Stewart STO-nG overlap,
/// shell-resolved Klopman–Ohno SCC, on-site third-order \(\Gamma q^3\),
/// exponential pair repulsion, and D3(BJ) dispersion with the GFN1 damping
/// set. Hydrogen is double-ζ (1s+2s, 2s orthogonalized). Static tables in
/// `basis_data/gfn1_tables.h` cover Z = 1, 6, 7, 8 only.
///
/// Energy foothold: `with_forces` defaults false and throws if set true
/// (analytic GFN forces are a later slug). Closed shell only. Units on the
/// public surface: energy [eV], positions [Å].
///
/// Oracle: **xtb** (never OpenMOPAC / DFTB+). Pin:
/// `regressions/semiempirical/gfn1-xtb-organic/`.
///
/// @code
///   cpu::Evaluate eval;
///   cpu::semiempirical::GFN1 gfn1;
///   gfn1.with_forces = false;
///   eval.on(evaluate::Phase::evaluate).add(std::move(gfn1));
/// @endcode
class GFN1 : public cpu::Semiempirical, public cpu::QmEngine {
  public:
    /// @brief SCC residual tolerance on max |Δq| over shells [e].
    double scc_tol = 1e-8;
    /// @brief Maximum SCC iterations (dimensionless). Exhausting it throws.
    int max_scc = 250;
    /// @brief Anderson charge-mixing relaxation β ∈ (0,1] (dimensionless).
    double mix_beta = 0.4;
    /// @brief Net molecular charge [e] (subtracted from valence electron count).
    int charge = 0;
    /// @brief Spin multiplicity 2S+1. Only 1 (closed shell) is supported.
    int multiplicity = 1;
    /// @brief When true, throw — nuclear forces are not a product path yet.
    bool with_forces = false;

    GFN1() = default;

    std::string name() const override { return "gfn1"; }

    bool provides_force() const override { return false; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    std::string qm_engine_name() const override { return name(); }
    void prepare(const BuildContext& qm_ctx) override { build(qm_ctx); }
    void evaluate(const LaunchContext& qm_ctx) override { launch(qm_ctx); }

    /// @brief Last total energy E_EHT+γ+Γ+rep+D3 [eV].
    double last_energy() const { return energy_; }
    /// @brief Last SCC residual max|Δq_shell| [e].
    double last_scc_residual() const { return scc_residual_; }
    /// @brief SCC iterations of the last solve.
    int last_scc_iterations() const { return scc_iterations_; }

  private:
    double energy_ = 0;
    double scc_residual_ = 0;
    int scc_iterations_ = 0;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV

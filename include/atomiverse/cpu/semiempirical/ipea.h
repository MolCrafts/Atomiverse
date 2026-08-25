/// @file cpu/semiempirical/ipea.h
/// @brief IPEA-xTB energy and analytic nuclear forces (organic H/C/N/O, CPU).
#pragma once

#include <string>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief IPEA-xTB total energy and analytic nuclear forces for H/C/N/O.
///
/// Special-purpose GFN1 reparametrization for ionization potentials and
/// electron affinities (Ásgeirsson, Bauer, Grimme, *Chem. Sci.* **8**, 4879
/// (2017), DOI 10.1039/c7sc00601b): EHT + Klopman–Ohno SCC with
/// \(g_{\exp}=1\), atomic third-order \(\Gamma q^3\), extra \((n+1)s\)
/// functions on C/N/O, exponential pair repulsion, D3(BJ) with the GFN1
/// damping set. Not GFN1, not GFN2, not DFTB. Static tables in
/// `basis_data/ipea_tables.h` cover Z = 1, 6, 7, 8 only.
///
/// Closed-form \(F=-\nabla E\) (Hellmann–Feynman + Pulay, CN/shell-poly
/// chains, \(\partial\gamma/\partial R\), pair repulsion, D3). Closed shell
/// only. Units on the public surface: energy [eV], positions [Å], forces
/// [eV/Å].
///
/// Oracle: **xtb** (never OpenMOPAC / DFTB+). Pins:
/// `regressions/semiempirical/ipea-xtb-small/` (energy),
/// `regressions/semiempirical/ipea-xtb-forces/` (forces).
///
/// @code
///   cpu::Evaluate eval;
///   cpu::semiempirical::IPEA ipea;
///   ipea.with_forces = true;
///   eval.on(evaluate::Phase::evaluate).add(std::move(ipea));
/// @endcode
class IPEA : public cpu::Semiempirical, public cpu::QmEngine {
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
    /// @brief When true, write analytic forces F=−∇E [eV/Å] on launch.
    bool with_forces = false;

    IPEA() = default;

    std::string name() const override { return "ipea"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    std::string qm_engine_name() const override { return name(); }
    void prepare(const BuildContext& qm_ctx) override { build(qm_ctx); }
    void evaluate(const LaunchContext& qm_ctx) override { launch(qm_ctx); }

    /// @brief Last total energy E_EHT+γ+Γ+rep+D3 [eV].
    double last_energy() const { return energy_; }
    /// @brief Last SCC residual max|Δq| [e].
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

/// @file cpu/scf/stages/fock_stage.h
/// @brief FockStage — the `scf::Phase::fock` stage component.
///
/// Wraps the shipping `internal::HamiltonianBuilder` (the method-dispatched
/// Fock build F = Hcore + J − αK + V_xc) as a config-free `cpu::Component`
/// bound at `scf::Phase::fock`, WITHOUT any driver wiring (that is sub-spec
/// 03). State flows through named ctor references — no config bag, no
/// workspace bag: the density inputs are `const&`, the Fock/Coulomb/energy
/// outputs are `&`. Method selection is a set of discrete constructor
/// parameters, passed straight to `HamiltonianBuilder`'s config-free entry, so
/// the outputs are bit-identical to the shipping solver path by construction.
/// Energy convention: Hartree throughout (matching the internal SCF helpers).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/hamiltonian_builder.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/pcm.h"
#include "atomiverse/cpu/scf/xc_functional.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Fock-build stage: F = Hcore + J − αK + V_xc, method-dispatched
///        (restricted/unrestricted × HF/KS) from discrete parameters.
///
/// Bit-identical to `internal::HamiltonianBuilder` on the same density inputs.
/// Bound at `scf::Phase::fock`; `launch()` self-guards via
/// `ctx.at(scf::Phase::fock)` and does nothing off any other stage.
class FockStage : public cpu::Component {
  public:
    /// @brief Construct the Fock stage from method parameters and wired state.
    /// @param restricted   True for closed-shell (α mirrors β).
    /// @param is_hf        True for Hartree-Fock; false for Kohn-Sham.
    /// @param xc_type      Exchange-correlation functional (KS only).
    /// @param grid_level   Named DFT integration-grid quality level (KS only).
    /// @param n_radial     Radial grid points per atom (SG0 only).
    /// @param n_angular    Angular grid points per atom (SG0 only).
    /// @param hybrid_alpha_override Regression override for the exact-exchange
    ///                     fraction; `< 0` uses the functional's own α_hf.
    /// @param fixed_grid   Optional externally supplied XC grid (KS only; not
    ///                     owned, must outlive this stage).
    /// @param integrals    Integral pack (S / Hcore / ERI / shells / grid
    ///                     inputs), caller-owned (read).
    /// @param P_alpha      α density (full closed-shell density when
    ///                     restricted) (read).
    /// @param P_beta       β density; alias of P_alpha when restricted (read).
    /// @param F_alpha      α Fock matrix, Hartree (write).
    /// @param F_beta       β Fock matrix, Hartree (write).
    /// @param J            Coulomb matrix built from the total density,
    ///                     Hartree (write).
    /// @param e_xc_grid    Grid-integrated E_xc (KS only), Hartree (write).
    /// @param e_exact_exchange Hybrid exact-exchange energy, Hartree (write).
    /// @param int_n_alpha  Grid-integrated α electron count (KS only) (write).
    /// @param int_n_beta   Grid-integrated β electron count (KS only) (write).
    /// @param ri           Density-fitting mode (RI::None = exact ERIs,
    ///                     bit-identical shipping path). Combining a
    ///                     non-None mode with `generalized` throws — the
    ///                     GHF spinor build contracts the stored dense
    ///                     ERI directly and has no RI wiring.
    /// @param aux_basis    Auxiliary basis for the fit (ignored when
    ///                     `ri == RI::None`).
    /// @param generalized  True for the GHF spinor build: `P_alpha` then
    ///                     carries the 2n×2n spinor density and `F_alpha`
    ///                     receives the 2n×2n spinor Fock
    ///                     F^{σσ'} = δ_{σσ'}(h + J[P^αα+P^ββ]) − K[P^{σσ'}]
    ///                     [Hartree], contracted per n×n spin block from the
    ///                     stored dense ERI (requires `integrals.ERI`;
    ///                     HF only).
    /// @param pcm          Optional non-owning C-PCM workspace (null = off).
    ///                     When set, each fock iteration updates the reaction
    ///                     field from the total density and adds V_pcm to both
    ///                     spin Focks.
    /// @param e_solvent    Electrostatic free energy ΔG_el = ½ q·V [Hartree]
    ///                     (write; 0 when pcm is null).
    /// @param e_pcm_fock_half Half-trace ½ Tr[V_pcm P_total] [Hartree] already
    ///                     folded into the HF energy ½ Tr[(H+F)P] when V_pcm
    ///                     sits on F — DensityStage subtracts it and adds
    ///                     `e_solvent` so the free-energy convention matches
    ///                     PySCF (write; 0 when pcm is null).
    FockStage(bool restricted, bool is_hf, XCType xc_type, GridLevel grid_level,
              int n_radial, int n_angular, double hybrid_alpha_override,
              const MolecularGrid* fixed_grid, const IntegralPack& integrals,
              const RealMatrix& P_alpha, const RealMatrix& P_beta,
              RealMatrix& F_alpha, RealMatrix& F_beta, RealMatrix& J,
              double& e_xc_grid, double& e_exact_exchange, double& int_n_alpha,
              double& int_n_beta, RI ri = RI::None,
              AuxBasis aux_basis = AuxBasis::Def2JKFIT,
              bool generalized = false, internal::Pcm* pcm = nullptr,
              double& e_solvent = FockStage::discard_double_(),
              double& e_pcm_fock_half = FockStage::discard_double_());

    /// @brief Component name.
    /// @return `"scf_fock_stage"`.
    std::string name() const override { return "scf_fock_stage"; }

    /// @brief Default binding: `scf::Phase::fock`, order 0.
    /// @return A single-element binding vector via `make_binding`.
    std::vector<DefaultBinding> default_bindings() const override;

    /// @brief One-time setup: build the shell-pair lists and (KS) the XC grid.
    /// @param ctx Build context (unused beyond lifecycle sequencing).
    void build(const cpu::BuildContext& ctx) override;

    /// @brief Build F/J and the KS energies from the wired densities.
    /// @param ctx Launch context; work runs only when `ctx.at(fock)`.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    /// Sink for optional default-bound double& parameters when the caller
    /// does not care about the PCM energy channels (vacuum path).
    static double& discard_double_() {
        static double sink = 0;
        return sink;
    }

    bool restricted_;
    bool is_hf_;
    bool generalized_; ///< GHF spinor build (2n×2n densities/Fock).
    XCType xc_type_;
    GridLevel grid_level_;
    int n_radial_;
    int n_angular_;
    double hybrid_alpha_override_;
    const MolecularGrid* fixed_grid_;
    RI ri_;              ///< Density-fitting mode (member-driven seam).
    AuxBasis aux_basis_; ///< Auxiliary basis for the fit.

    const IntegralPack& integrals_;
    const RealMatrix& P_alpha_;
    const RealMatrix& P_beta_;

    RealMatrix& F_alpha_;
    RealMatrix& F_beta_;
    RealMatrix& J_;
    double& e_xc_grid_;
    double& e_exact_exchange_;
    double& int_n_alpha_;
    double& int_n_beta_;

    internal::Pcm* pcm_ = nullptr; ///< Non-owning; null = vacuum.
    double& e_solvent_;
    double& e_pcm_fock_half_;

    internal::HamiltonianBuilder hamiltonian_;
};

} // namespace scf
} // namespace cpu
} // namespace ATV

/// @file cpu/scf/stages/density_stage.h
/// @brief DensityStage — the `scf::Phase::density` stage component.
///
/// Wraps the shipping solver's density-build + damping + energy block
/// (`scf_solver.cpp` step 4-6) as a config-free `cpu::Component` bound at
/// `scf::Phase::density`, WITHOUT any driver wiring (that is sub-spec 03).
/// State flows through named ctor references — no config struct, no workspace
/// bag: reads are `const&`, the density and energy outputs are `&`.
///
/// The `P_alpha` / `P_beta` references are **old-in / new-out**: on entry they
/// carry the previous iteration's density (read by the HF energy branches);
/// on exit they carry the freshly built, optionally damped, density. Energy
/// convention: Hartree throughout (matching the internal SCF helpers).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/density_builder.h"
#include "atomiverse/cpu/scf/scf_enums.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Density-build stage: `P_new = factor·C_occ·C_occᵀ`, damping toward
///        the retained old density (when DIIS is inactive), and the
///        four-branch (restricted/unrestricted × HF/KS) energy evaluation.
///
/// Bit-identical to `scf_solver.cpp` steps 4-6. Bound at `scf::Phase::density`;
/// `launch()` self-guards via `ctx.at(scf::Phase::density)`.
class DensityStage : public cpu::Component {
  public:
    /// @brief Construct the density stage from method flags and wired state.
    /// @param restricted        True for closed-shell (α mirrors β).
    /// @param is_hf             True for Hartree-Fock (energy uses the OLD
    ///                          density); false for Kohn-Sham (NEW density).
    /// @param damping           Fraction of the OLD density retained when DIIS
    ///                          is inactive (PySCF convention), dimensionless.
    /// @param C_alpha           α MO coefficients (read).
    /// @param C_beta            β MO coefficients (read).
    /// @param F_alpha           α Fock matrix, Hartree (read; HF energy only).
    /// @param F_beta            β Fock matrix, Hartree (read; HF energy only).
    /// @param Hcore             Core Hamiltonian, Hartree (read).
    /// @param J                 Coulomb matrix, Hartree (read; KS energy only).
    /// @param n_alpha           Occupied α orbital count (read).
    /// @param n_beta            Occupied β orbital count (read).
    /// @param n_basis           Basis dimension (read).
    /// @param diis_active       True once DIIS is extrapolating (suppresses
    ///                          damping) (read).
    /// @param e_xc_grid         Grid XC energy, Hartree (read; KS energy only).
    /// @param e_exact_exchange  Hybrid exact-exchange energy, Hartree (read;
    ///                          KS energy only).
    /// @param v_nuc             Nuclear-repulsion energy, Hartree (read).
    /// @param P_alpha           α density: OLD in / NEW out (read then write).
    /// @param P_beta            β density: OLD in / NEW out (read then write).
    /// @param e_total           Total SCF energy, Hartree (write).
    /// @param e_one             One-electron energy, Hartree (write).
    /// @param e_coulomb         Coulomb (J) energy, Hartree (write).
    /// @param generalized       True for the GHF spinor path: `C_alpha` is
    ///                          the 2n×2n_mo spinor coefficient matrix,
    ///                          `n_alpha` the total spinor occupation N,
    ///                          `P_alpha` the 2n×2n spinor density, `Hcore`
    ///                          the spinor core Hamiltonian h⊗1₂ [Hartree];
    ///                          the energy is the single-channel
    ///                          E = ½Tr[P(H+F)] + E_nn on the OLD density.
    /// @param e_solvent         C-PCM free energy ΔG_el [Hartree] (read; 0
    ///                          when solvent is off). Added to every energy
    ///                          branch; for HF the half-trace of V_pcm on F
    ///                          is subtracted first via `e_pcm_fock_half`.
    /// @param e_pcm_fock_half   ½ Tr[V_pcm P_total] [Hartree] already inside
    ///                          the HF ½ Tr[(H+F)P] formula (read; 0 off).
    /// @param S                 AO overlap (dimensionless); used only when
    ///                          `mom_style != Off` for projection scores.
    /// @param mom_style         Maximum-overlap occupation selector (read).
    /// @param mom_ref_C_alpha   Reference occupied α MO columns [n × n_α]
    ///                          (read/write: updated each step under MOM).
    /// @param mom_ref_C_beta    Reference occupied β MO columns [n × n_β].
    DensityStage(
        bool restricted, bool is_hf, double damping, const RealMatrix& C_alpha,
        const RealMatrix& C_beta, const RealMatrix& F_alpha,
        const RealMatrix& F_beta, const RealMatrix& Hcore, const RealMatrix& J,
        const int& n_alpha, const int& n_beta, const int& n_basis,
        const bool& diis_active, const double& e_xc_grid,
        const double& e_exact_exchange, const double& v_nuc,
        RealMatrix& P_alpha, RealMatrix& P_beta, double& e_total, double& e_one,
        double& e_coulomb, bool generalized = false,
        const double& e_solvent = DensityStage::discard_double_(),
        const double& e_pcm_fock_half = DensityStage::discard_double_(),
        const RealMatrix& S = DensityStage::discard_matrix_(),
        const MomStyle& mom_style = DensityStage::discard_mom_(),
        RealMatrix& mom_ref_C_alpha = DensityStage::discard_matrix_mut_(),
        RealMatrix& mom_ref_C_beta = DensityStage::discard_matrix_mut_());

    /// @brief Component name.
    /// @return `"scf_density_stage"`.
    std::string name() const override { return "scf_density_stage"; }

    /// @brief Default binding: `scf::Phase::density`, order 0.
    /// @return A single-element binding vector via `make_binding`.
    std::vector<DefaultBinding> default_bindings() const override;

    /// @brief Pre-allocate the density-builder workspace (no launch alloc).
    /// @param ctx Build context (unused beyond lifecycle sequencing).
    void build(const cpu::BuildContext& ctx) override;

    /// @brief Build the new density, damp, and evaluate the branch energy.
    /// @param ctx Launch context; work runs only when `ctx.at(density)`.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    static const double& discard_double_() {
        static double sink = 0;
        return sink;
    }
    static const RealMatrix& discard_matrix_() {
        static RealMatrix sink;
        return sink;
    }
    static RealMatrix& discard_matrix_mut_() {
        static RealMatrix sink;
        return sink;
    }
    static const MomStyle& discard_mom_() {
        static MomStyle sink = MomStyle::Off;
        return sink;
    }

    /// @brief Build P = factor · Σ_{j∈S} c_j c_jᵀ where S is the n_occ
    ///        new MOs with largest projection onto `C_ref` under metric S.
    ///        When style is MOM, overwrite `C_ref` with the selected columns.
    static RealMatrix mom_density_(const RealMatrix& C, RealMatrix& C_ref,
                                   const RealMatrix& S, int n_occ, int n_basis,
                                   double factor, MomStyle style);

    bool restricted_;
    bool is_hf_;
    bool generalized_; ///< GHF spinor path (2n dimensions, single channel).
    double damping_;

    const RealMatrix& C_alpha_;
    const RealMatrix& C_beta_;
    const RealMatrix& F_alpha_;
    const RealMatrix& F_beta_;
    const RealMatrix& Hcore_;
    const RealMatrix& J_;
    const RealMatrix& S_;

    const int& n_alpha_;
    const int& n_beta_;
    const int& n_basis_;
    const bool& diis_active_;
    const double& e_xc_grid_;
    const double& e_exact_exchange_;
    const double& v_nuc_;
    const double& e_solvent_;
    const double& e_pcm_fock_half_;
    const MomStyle& mom_style_;

    RealMatrix& P_alpha_;
    RealMatrix& P_beta_;
    RealMatrix& mom_ref_C_alpha_;
    RealMatrix& mom_ref_C_beta_;
    double& e_total_;
    double& e_one_;
    double& e_coulomb_;

    internal::DensityBuilder density_;
};

} // namespace scf
} // namespace cpu
} // namespace ATV

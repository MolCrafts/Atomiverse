/// @file cpu/scf/internal/hamiltonian_builder.h
/// @brief internal::HamiltonianBuilder — F = Hcore + J − αK + V_xc.
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component`.
///
/// Encapsulates the method-dispatched Fock build (RHF / UHF / RKS / UKS)
/// + ownership of the precomputed shell-pair lists and integration grid.
/// Owned by the concrete method; `prepare()` runs once per molecular-shape
/// (n_basis fixed) and `step()` is reused across all SCF iterations.
#pragma once

#include <memory>
#include <span>

#include "atomiverse/cpu/scf/basis_set.h" // AuxBasis
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/cfmm_j.h"
#include "atomiverse/cpu/scf/internal/density_fitting.h"
#include "atomiverse/cpu/scf/internal/exact_jk.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/scf/scf_enums.h" // MeanFieldType / SpinType / RI / Coulomb
#include "atomiverse/cpu/scf/xc_grid_engine.h" // XCGridBasis (per-geometry cache)
#include "atomiverse/cpu/xc/xc_functional.h"   // xc::XCType

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

struct FockResult {
    RealMatrix F_alpha;     ///< α-channel Fock matrix [n × n].
    RealMatrix F_beta;      ///< β-channel Fock matrix [n × n].
    RealMatrix J;           ///< Coulomb J built from the total density.
    double E_xc_grid = 0;   ///< Grid-integrated E_xc (KS only) [Hartree].
    double int_n_alpha = 0; ///< Grid-integrated α electron count (KS only).
    double int_n_beta = 0;  ///< Grid-integrated β electron count (KS only).
    /// @brief Exact (Hartree-Fock) exchange energy admixture for a KS
    ///        global hybrid: `alpha_hf * E_K` [Hartree]. Zero for pure
    ///        KS and for HF (HF carries its exchange inside F directly).
    ///        The Fock builder applies `alpha_hf` at a single site so this
    ///        term and the `-alpha_hf*K` Fock term use the same scaled K.
    double E_exact_exchange = 0;
};

class HamiltonianBuilder {
  public:
    HamiltonianBuilder() = default;

    /// @brief One-time setup from discrete method parameters — the entry used
    ///        by the driver-side `FockStage`.
    /// @details Builds shell-pair lists and (for KS) the XC integration grid.
    /// @param integrals Non-owning view of method-owned integral fields.
    /// @param is_hf     true → Hartree-Fock (no grid); false → Kohn-Sham
    ///                  (builds the XC integration grid).
    /// @param grid_level Named DFT integration-grid quality level (KS only).
    /// @param n_radial   Radial grid points per atom (SG0 only).
    /// @param n_angular  Angular grid points per atom (SG0 only).
    /// @param fixed_grid Optional external grid (KS only; not owned).
    /// @param ri         Density-fitting mode; `RI::None` (default) keeps
    ///                   the exact-ERI path bit-identical. Non-None builds
    ///                   the `internal::DensityFitting` workspace over the
    ///                   method's Cartesian shells (this seam is reused by
    ///                   the CFMM link).
    /// @param aux_basis  Auxiliary basis for the fit (ignored when
    ///                   `ri == RI::None`); an element without a table row
    ///                   throws — no fallback.
    /// @param coulomb    J-matrix algorithm (`Coulomb::Direct` default =
    ///                   bit-identical vacuum path). `Coulomb::CFMM` builds
    ///                   the `internal::CfmmJ` workspace; combining CFMM with
    ///                   `ri != RI::None` throws (J route is exclusive).
    /// @param cfmm_order Multipole truncation order p for CFMM (default 10).
    /// @param cfmm_threshold Extent threshold ε_thr for CFMM (default 1e-9).
    void prepare(const internal::IntegralView& integrals, bool is_hf,
                 GridLevel grid_level, int n_radial, int n_angular,
                 const MolecularGrid* fixed_grid, RI ri = RI::None,
                 AuxBasis aux_basis = AuxBasis::Def2JKFIT,
                 Coulomb coulomb = Coulomb::Direct, int cfmm_order = 10,
                 double cfmm_threshold = 1e-9, double max_memory_mb = 4000.0,
                 bool mpi = false);

    /// Exact J/K in the working AO basis for several densities in one
    /// integral pass, also for nonsymmetric spin blocks (incremental per
    /// request channel).
    void exact_jk(std::span<const JKRequest> requests) const {
        exact_.step(requests);
    }

    [[nodiscard]] const ExactJK& coulomb_engine() const { return exact_; }

    /// @brief Build F from the current α/β densities.
    /// @details Method-dispatched F = Hcore + J − αK + V_xc.
    /// @param integrals Same IntegralView passed to prepare() (refs still live).
    /// @param restricted true → restricted spin (α mirrors β); false → open.
    /// @param is_hf      true → Hartree-Fock; false → Kohn-Sham.
    /// @param xc_type    Exchange-correlation functional (KS only).
    /// @param hybrid_alpha_override Regression override for the exact-exchange
    ///                   fraction; `< 0` uses the functional's own α_hf.
    /// @param P_alpha    α density (full density when restricted).
    /// @param P_beta     β density (alias of P_alpha when restricted).
    /// @return Filled FockResult.
    FockResult
    step(const internal::IntegralView& integrals, bool restricted, bool is_hf,
         xc::XCType xc_type, double hybrid_alpha_override,
         const RealMatrix& P_alpha, const RealMatrix& P_beta,
         const ATV::electronic::XcFunctional* functional = nullptr) const;

  private:
    /// The exact engine for operator 1/r (omega = 0) or erf(omega r)/r,
    /// prepared on first use when prepare() did not need it.
    const ExactJK& exact_engine_(const internal::IntegralView& integrals,
                                 double omega) const;

    /// Exact 1/r J/K: every Direct/CFMM Coulomb build and all exact exchange.
    /// `mutable` so the RI::J exchange engine can be prepared on first use.
    mutable ExactJK exact_;
    /// Exact erf(mu r)/r exchange of a range-separated hybrid (lazy).
    mutable ExactJK exact_lr_;
    double memory_mb_ = 4000.0; ///< ExactJK cache budget from prepare().
    bool mpi_ = false;          ///< ExactJK MPI distribution from prepare().
    MolecularGrid grid_;        ///< KS XC grid (empty for HF).
    /// Geometry-fixed, block-compacted AO-on-grid basis (values, and
    /// gradients for GGA), rebuilt only per geometry: `prepare()` invalidates
    /// it (built=false) and the first KS `step()` lazily builds it, so the
    /// transcendental basis evaluations are hoisted out of the SCF iteration
    /// loop. `mutable` because `step()` is const. Memory is the screened
    /// block product (see xc_grid_engine.h).
    mutable XCGridBasis xc_basis_;
    RI ri_ = RI::None; ///< Density-fitting mode set by prepare().
    Coulomb coulomb_ = Coulomb::Direct; ///< J-matrix algorithm.
    /// RI workspace ((μν|P), metric Cholesky, B tensor); null when
    /// `ri_ == RI::None` — the exact path never touches it.
    std::unique_ptr<DensityFitting> df_;
    /// CFMM-J workspace; null unless `coulomb_ == Coulomb::CFMM`.
    std::unique_ptr<CfmmJ> cfmm_;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

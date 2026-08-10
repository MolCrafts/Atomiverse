/// @file cuda/gpw/gpw_solver.cuh
/// @brief `cuda::gpw::GpwDft` — concrete GPW DFT method on CUDA (first product).
///
/// Public twin of `cpu::gpw::GpwDft` for the **product energy slice**:
/// Γ-point closed-shell DIIS RKS with Padé LDA or PBE-GGA, single-grid
/// (`ngrids==1`) **or** REL_CUTOFF multigrid (`ngrids>1` + `rel_cutoff>0`),
/// periodic Poisson. Parameters are public members (no `*Config` bag, no
/// `make_*` factory). Stage footholds under `cuda/gpw/internal/` are composed
/// inside `run()` and are never the user API.
///
/// Hybrid composition (stated, not silent):
///   * host — STVnl / product-list build / DIIS / diagonalize / acc scatter
///   * device — product collocate + KS + product integrate via
///     `internal::KsChainDevice::run` (single-grid) or
///     `internal::KsMultiGridChainDevice::run` (multigrid: per-level collocate,
///     spectral prolong, fine Poisson/XC, spectral restrict, per-level
///     integrate). Mid-SCF does not H2D ρ or D2H V_eff for H.
///
/// Unsupported members throw with a how-to (no silent CPU-SCF fallback).
/// Contract: `.claude/notes/cuda-gpw-public-shape.md`.
///
/// @code
///   cuda::gpw::GpwDft m;
///   m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
///   m.cutoff_ry = 100.0;
///   m.n_occ = 1;
///   m.xc = cpu::gpw::XcKind::PADE;
///   m.build(atoms);
///   const auto& res = m.run(atoms);   // energy [Hartree]
///   auto F = cuda::gpw::gpw_forces_analytic(atoms, m);
/// @endcode
#pragma once

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/linalg/matrix.h"

#include <array>
#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {

/// @brief Result type shared verbatim with the CPU method (energy in Hartree).
using GpwResult = ::ATV::cpu::gpw::GpwResult;
/// @brief XC selector shared with the CPU method.
using XcKind = ::ATV::cpu::gpw::XcKind;
/// @brief Atom type shared with the CPU method.
using GpwAtom = ::ATV::cpu::gpw::GpwAtom;

/// @brief Concrete GPW DFT method on CUDA — members + `build` / `run` /
///        `last_result`.
///
/// Product energy: Γ RKS DIIS, `XcKind::PADE` or `XcKind::PBE`, single grid
/// or REL_CUTOFF multigrid, periodic, fixed occupations. Move-only.
class GpwDft {
  public:
    // ── Method parameters (mirror CPU GpwDft product energy members) ─────
    double cell[3] = {0, 0, 0}; ///< orthorhombic box lengths [Bohr].
    double cutoff_ry = 300.0;   ///< density grid cutoff [Ry].
    int ngrids = 1;             ///< multigrid levels (≥1; >1 needs rel_cutoff>0).
    double rel_cutoff = 0.0;    ///< CP2K REL_CUTOFF [Ry] (required when ngrids>1).
    int n_occ = 0;              ///< RKS: doubly-occupied orbitals (N_elec / 2).
    int n_alpha = 0;            ///< UKS α (device product: must stay 0).
    int n_beta = 0;             ///< UKS β (device product: must stay 0).
    int max_iter = 100;         ///< max SCF iterations.
    double tol = 1e-8;          ///< SCF energy convergence [Hartree].
    double eps_default = 1e-8;  ///< Gaussian-tail cutoff.
    double mix = 0.5;           ///< linear density mixing when diis_depth==0.
    int diis_depth = 6;         ///< Pulay DIIS history depth.
    double lindep_threshold = 1e-8; ///< S-eigenvalue keep threshold.
    double level_shift = 0.0;       ///< virtual level shift [Hartree].
    XcKind xc = XcKind::PADE;       ///< PADE or PBE on device.
    bool nonperiodic = false;       ///< MT Poisson — not on device (throws).
    bool local_pp_on_grid = true;   ///< collocate GTH local SR PP on grid.
    bool use_ot = false;            ///< OT — not on device product (throws).
    int k_mesh[3] = {1, 1, 1};      ///< must stay Γ for this product depth.

    GpwDft() = default;
    ~GpwDft() = default;
    GpwDft(const GpwDft&) = delete;
    GpwDft& operator=(const GpwDft&) = delete;
    GpwDft(GpwDft&&) noexcept = default;
    GpwDft& operator=(GpwDft&&) noexcept = default;

    /// @brief One-time fail-fast validation against a reference system.
    /// @param reference Atoms the solver will be run on (structural only).
    /// @throws std::invalid_argument on empty atoms, missing RKS occupation,
    ///         non-positive cell edges, or first-slice-unsupported members.
    void build(const std::vector<GpwAtom>& reference);

    /// @return true when configured for UKS (`n_alpha+n_beta>0`).
    bool is_uks() const { return n_alpha + n_beta > 0; }

    /// @brief Γ RKS DIIS SCF at the given geometry (device KS, single- or multi-grid).
    ///
    /// Re-entrant: geometry-dependent structures are rebuilt per call.
    /// Multigrid (`ngrids>1 && rel_cutoff>0`) collocates products per level,
    /// prolongs to the fine grid for Poisson/XC, restricts V_eff, and integrates
    /// per level — twin of CPU `multigrid_collocate_rho` / integrate.
    /// @param atoms Atom set at the current geometry [Bohr].
    /// @param guess_C Optional warm-start MO coefficients (nb × nb).
    /// @param guess_P Optional initial closed-shell density (nb × nb).
    /// @return Owned result (valid until the next `run()`).
    /// @throws std::logic_error when called before `build()`.
    /// @throws std::invalid_argument for unsupported members / bad guesses.
    GpwResult& run(const std::vector<GpwAtom>& atoms,
                   const cpu::linalg::RealMatrix* guess_C = nullptr,
                   const cpu::linalg::RealMatrix* guess_P = nullptr);

    /// @brief Result of the most recent `run()` (`converged==false` before first).
    const GpwResult& last_result() const { return result_; }

  private:
    void validate_slice_() const;
    GpwResult result_{};
    bool built_ = false;
};

/// @brief Analytic ionic forces after a CUDA Γ RKS SCF (free-function family).
///
/// Runs `method.run(atoms)` then assembles forces from the converged
/// `GpwResult` via the same analytic path as CPU `gpw_forces_from_scf`
/// (Pulay + core + grid HF on the SCF V_eff / V_es). Forces match the density
/// that produced `method.last_result().e_total`.
///
/// @param atoms Geometry + basis/PP [Bohr].
/// @param method CUDA method (members must match the SCF that will run).
/// @return Forces [Hartree/Bohr], one xyz triple per atom.
/// @throws std::runtime_error if the SCF does not converge.
std::vector<std::array<double, 3>>
gpw_forces_analytic(const std::vector<GpwAtom>& atoms, GpwDft& method);

/// @brief Analytic forces from a **converged** CUDA `GpwResult` (no second SCF).
///
/// @param atoms Geometry + basis/PP [Bohr].
/// @param method Method whose members match the SCF that produced `scf`.
/// @param scf Converged result (`converged` required).
/// @return Forces [Hartree/Bohr].
std::vector<std::array<double, 3>>
gpw_forces_from_scf(const std::vector<GpwAtom>& atoms, const GpwDft& method,
                    const GpwResult& scf);

} // namespace gpw
} // namespace cuda
} // namespace ATV

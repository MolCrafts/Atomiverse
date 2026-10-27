/// @file cuda/gpw/gpw_solver.cuh
/// @brief `cuda::gpw::GpwDft` — concrete GPW DFT method on CUDA (product).
///
/// Public twin of `cpu::gpw::GpwDft` for the **landed product slice**:
/// Γ-point DIIS RKS (Padé LDA or PBE-GGA) with single-grid (`ngrids==1`) **or**
/// REL_CUTOFF multigrid (`ngrids>1` + `rel_cutoff>0`), **or collinear UKS
/// (Padé LSD, single-grid only)**, **or multi-k RKS DIIS** (single-grid,
/// Monkhorst–Pack / explicit list; device phased STVnl + host χ^k,
/// device Poisson/XC), periodic Poisson. Parameters are public members (no
/// `*Config` bag, no `make_*` factory). Stage footholds under
/// `cuda/gpw/internal/` are composed inside `run()` and are never the user API.
///
/// Hybrid composition (stated, not silent):
///   * host — Γ STVnl / product-list build (once); multi-k χ^k build once;
///     multi-k residual: energy ctrace pieces / payload diag / cold-start;
///     force AO Pulay (W·dS / P·dT / P·dVnl) on Γ free-fn path
///   * device mid-SCF — SG: product pack→PBC collocate→KS→integrate→scatter;
///     MG: monom multi-level pack→`KsMultiGridChainDevice`→scatter (host collint
///     only if `ATV_GPW_CUDA_HOST_MG=1`); UKS: dual pack + `run_uks`; multi-k:
///     device STVnl S/T, χ collocate, `run_from_rho`, V→H_k, H0+V fuse, H
///     residency into DIIS, `KpointDiisEigenDevice`. Γ RKS/UKS: device H=H0+V,
///     `AoDiisEigenDevice` DIIS/eigen/density, final `eigen_from_device`
///     payload (no host re-diag). Non-default stream for SCF body.
///
/// Unsupported members throw with a how-to (no silent CPU-SCF fallback).
/// Contract: `.claude/notes/cuda-gpw-public-shape.md`.
///
/// @code
///   cuda::gpw::GpwDft m({{el, basis, pot}}, /*cutoff_ry=*/100.0,
///                       cpu::xc::XCType::PADE);
///   m.n_occ = 5;                   // or n_alpha / n_beta for UKS
///   m.build(frame);                // cell from frame.box, Z → kinds
///   const auto& res = m.run(frame); // kEnergy [eV]; result in Hartree
///   m.forces(frame);               // kAtom_f* [eV/Å]
/// @endcode
#pragma once

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/linalg/matrix.h"

#include <array>
#include <utility>
#include <vector>

namespace ATV {
namespace cuda {
namespace gpw {

/// @brief Result type shared verbatim with the CPU method (energy in Hartree).
using GpwResult = ::ATV::cpu::gpw::GpwResult;
/// @brief Atom type shared with the CPU method.
using GpwAtom = ::ATV::cpu::gpw::internal::GpwAtom;

/// @brief Concrete GPW DFT method on CUDA — members + `build` / `run` /
///        `last_result`.
///
/// Product slice: Γ DIIS RKS (`n_occ`, PADE/PBE, single grid or REL_CUTOFF
/// multigrid), collinear UKS (`n_alpha`/`n_beta`, PADE LSD, single grid), or
/// multi-k RKS DIIS (`k_mesh` / `k_points`, single grid, fixed occ), periodic.
/// Move-only.
class GpwDft {
  public:
    // ── Identity (constructor arguments, same as `cpu::gpw::GpwDft`) ─────
    /// @brief Per-element basis + GTH potential. `build(Frame)` / `run(Frame)`
    ///        map `kAtom_type` (Z) onto these entries. The atom-list entry
    ///        points do not read this member.
    const std::vector<cpu::gpw::Kind> kinds;
    const double cutoff_ry;   ///< density grid cutoff [Ry].
    const cpu::xc::XCType xc; ///< RKS: PADE or PBE; UKS: PADE only (first cut).

    // ── Knobs ──────────────────────────────────────────────────────────
    double cell[3] = {0, 0, 0}; ///< orthorhombic box lengths [Bohr].
    int ngrids = 1;          ///< multigrid levels (≥1; >1 needs rel_cutoff>0).
    double rel_cutoff = 0.0; ///< CP2K REL_CUTOFF [Ry] (required when ngrids>1).
    int n_occ = 0;           ///< RKS: doubly-occupied orbitals (N_elec / 2).
    int n_alpha = 0;         ///< UKS α occupations (with `n_beta`; 0 ⇒ RKS).
    int n_beta = 0;          ///< UKS β occupations (with `n_alpha`; 0 ⇒ RKS).
    int max_iter = 100;      ///< max SCF iterations.
    double tol = 1e-8;       ///< SCF energy convergence [Hartree].
    double eps_default = 1e-8; ///< Gaussian-tail cutoff.
    double mix = 0.5;          ///< linear density mixing when diis_depth==0.
    int diis_depth = 6;        ///< Pulay DIIS history depth.
    double lindep_threshold = 1e-8; ///< S-eigenvalue keep threshold.
    double level_shift = 0.0;       ///< virtual level shift [Hartree] (Γ path).
    bool nonperiodic = false;       ///< MT Poisson — not on device (throws).
    bool local_pp_on_grid = true;   ///< collocate GTH local SR PP on grid.
    bool use_ot = false;            ///< OT — not on device product (throws).
    /// @brief Monkhorst–Pack subdivisions (CP2K convention). `{1,1,1}` is Γ
    ///        product path; any \(N_d>1\) selects multi-k RKS DIIS (single-grid).
    int k_mesh[3] = {1, 1, 1};
    /// @brief Explicit fractional k list (overrides `k_mesh` when non-empty).
    std::vector<std::array<double, 3>> k_points;
    /// @brief Weights for `k_points` (must match length; Σ = 1).
    std::vector<double> k_point_weights;
    /// @brief Time-reversal fold of the Monkhorst–Pack mesh (CPU twin default).
    bool k_time_reversal = true;

    /// @brief GPW Kohn–Sham method (same signature as `cpu::gpw::GpwDft`).
    /// @param kinds One basis + GTH entry per element in the Frames to solve.
    /// @param cutoff_ry Density grid cutoff [Ry].
    /// @param xc Exchange–correlation functional.
    GpwDft(std::vector<cpu::gpw::Kind> kinds, double cutoff_ry,
           cpu::xc::XCType xc)
        : kinds(std::move(kinds)), cutoff_ry(cutoff_ry), xc(xc) {}
    ~GpwDft() = default;
    GpwDft(const GpwDft&) = delete;
    GpwDft& operator=(const GpwDft&) = delete;
    GpwDft(GpwDft&&) noexcept = default;
    GpwDft& operator=(GpwDft&&) = delete;

    /// @brief One-time fail-fast validation against a reference system.
    /// @param reference Atoms the solver will be run on (structural only).
    /// @throws std::invalid_argument on empty atoms, missing occupation,
    ///         non-positive cell edges, or product-slice-unsupported members.
    void build(const std::vector<GpwAtom>& reference);

    /// @brief Validate against a Frame and copy its orthorhombic cell.
    ///
    /// Sets `cell` from `frame.box` [Å → Bohr]. Each atom's Z must match one
    /// `kinds` entry. Closed-shell RKS with `n_occ == 0` fills `n_occ` from
    /// the valence electron count.
    /// @param reference Geometry [Å] + Z.
    /// @throws std::invalid_argument on an empty Frame, a sheared cell, a Z
    ///         with no `Kind`, or an odd valence count when `n_occ` is unset.
    void build(const cpu::Frame& reference);

    /// @return true when configured for UKS (`n_alpha+n_beta>0`).
    bool is_uks() const { return n_alpha + n_beta > 0; }

    /// @return true when multi-k sampling is requested (non-Γ mesh or explicit list).
    bool uses_kpath() const {
        return !k_points.empty() || k_mesh[0] != 1 || k_mesh[1] != 1
               || k_mesh[2] != 1;
    }

    /// @brief DIIS SCF at the given geometry (device KS).
    ///
    /// RKS when `n_occ` set; collinear UKS when `n_alpha`/`n_beta` set;
    /// multi-k RKS when `uses_kpath()`. Re-entrant: geometry-dependent
    /// structures are rebuilt per call. Multigrid (`ngrids>1 && rel_cutoff>0`)
    /// is Γ RKS-only. UKS and multi-k require single-grid. Multi-k energy:
    /// device χ collocate from resident \(P_k\) + `run_from_rho` (χ tables
    /// built on host once).
    /// @param atoms Atom set at the current geometry [Bohr].
    /// @param guess_C Optional warm-start MO coefficients (nb × nb) — Γ only.
    /// @param guess_P Optional initial density (RKS closed-shell, or UKS α) — Γ.
    /// @param guess_P_beta Optional UKS β density (required with `guess_P` on UKS).
    /// @return Owned result (valid until the next `run()`).
    /// @throws std::logic_error when called before `build()`.
    /// @throws std::invalid_argument for unsupported members / bad guesses.
    GpwResult& run(const std::vector<GpwAtom>& atoms,
                   const cpu::linalg::RealMatrix* guess_C = nullptr,
                   const cpu::linalg::RealMatrix* guess_P = nullptr,
                   const cpu::linalg::RealMatrix* guess_P_beta = nullptr);

    /// @brief DIIS SCF at the Frame geometry. Does not write the Frame.
    /// @param frame Geometry [Å] + Z. `cell` is refreshed from `frame.box`.
    /// @param guess_C Optional Γ MO warm-start.
    /// @param guess_P Optional Γ density warm-start.
    /// @param guess_P_beta Optional UKS β density.
    /// @return Owned result (Hartree), valid until the next `run()`.
    /// @throws std::logic_error when called before `build()`.
    GpwResult& run(const cpu::Frame& frame,
                   const cpu::linalg::RealMatrix* guess_C = nullptr,
                   const cpu::linalg::RealMatrix* guess_P = nullptr,
                   const cpu::linalg::RealMatrix* guess_P_beta = nullptr);

    /// @brief `run(const Frame&)`, then write `kEnergy` [eV].
    /// @param frame Geometry in, energy meta out.
    /// @return Owned result (Hartree).
    GpwResult& run(cpu::Frame& frame,
                   const cpu::linalg::RealMatrix* guess_C = nullptr,
                   const cpu::linalg::RealMatrix* guess_P = nullptr,
                   const cpu::linalg::RealMatrix* guess_P_beta = nullptr);

    /// @brief SCF plus analytic forces into `kAtom_f*` [eV/Å] and `kEnergy` [eV].
    /// @param frame Geometry in, forces and energy out.
    /// @return The SCF result that produced the forces.
    /// @throws std::runtime_error if the SCF does not converge.
    GpwResult& forces(cpu::Frame& frame);

    /// @brief Result of the most recent `run()` (`converged==false` before first).
    const GpwResult& last_result() const { return result_; }

  private:
    void validate_slice_() const;
    GpwResult result_{};
    bool built_ = false;
};

/// @brief Analytic ionic forces after a CUDA Γ SCF (free-function family).
///
/// Runs `method.run(atoms)` then assembles forces from the converged
/// `GpwResult`: Γ RKS single-grid uses host AO Pulay + device grid HF.
/// UKS / multi-k / multigrid assemble on the host at the CUDA density
/// payload (same hybrid as `gpw_stress`). Forces match the density that
/// produced `method.last_result().e_total`.
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

/// @brief Analytic cell stress after a CUDA Γ SCF (free-function family).
///
/// Runs `method.run(atoms)` then assembles fixed-\(P\) orthorhombic stress
/// from the converged `GpwResult` (first cut: host analytic twin via
/// `cpu::gpw::gpw_stress` at the CUDA density payload — same shape as CPU
/// free-fn stress). Voigt order xx,yy,zz,xy,xz,yz [Hartree/Bohr³].
/// Γ / UKS / multi-k / multigrid use the same host fixed-\(P\) path when
/// energy is owned; deeper device stress assembly is residual.
///
/// @param atoms Geometry + basis/PP [Bohr].
/// @param method CUDA method (members must match the SCF that will run).
/// @param h Dimensionless strain step for fixed-\(P\) energy FD (default 2e-3).
/// @param axes Per-axis mask (x,y,z); masked axes report exactly 0.
/// @return Voigt stress [Hartree/Bohr³].
/// @throws std::runtime_error if the SCF does not converge.
std::array<double, 6>
gpw_stress(const std::vector<GpwAtom>& atoms, GpwDft& method, double h = 2e-3,
           const std::array<bool, 3>& axes = {true, true, true});

/// @brief Analytic cell stress from a **converged** CUDA `GpwResult` (no second SCF).
///
/// @param atoms Geometry + basis/PP [Bohr].
/// @param method Method whose members match the SCF that produced `scf`.
/// @param scf Converged result (`converged` required).
/// @param h Dimensionless strain step (default 2e-3).
/// @param axes Per-axis mask (x,y,z).
/// @return Voigt stress [Hartree/Bohr³].
std::array<double, 6>
gpw_stress(const std::vector<GpwAtom>& atoms, const GpwDft& method,
           const GpwResult& scf, double h = 2e-3,
           const std::array<bool, 3>& axes = {true, true, true});

} // namespace gpw
} // namespace cuda
} // namespace ATV

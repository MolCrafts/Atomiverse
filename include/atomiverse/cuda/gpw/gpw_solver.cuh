/// @file cuda/gpw/gpw_solver.cuh
/// @brief `cuda::gpw::GpwDft` — concrete GPW DFT method on CUDA (product).
///
/// Public twin of `cpu::gpw::GpwDft` for the **landed product slice**:
/// Γ-point DIIS RKS (Padé LDA or pure PBE-family GGA: PBE, PBEsol, RevPBE,
/// RPBE) with single-grid (`ngrids==1`) **or**
/// REL_CUTOFF multigrid (`ngrids>1` + `rel_cutoff>0`), **or collinear UKS
/// (Padé LSD or PBE-family spin GGA, single-grid only)**, **or multi-k RKS DIIS** (fixed occupations, single-grid or multigrid,
/// Monkhorst–Pack / explicit list; device phased STVnl and Poisson/XC), periodic Poisson. Parameters are public members (no
/// `*Config` bag, no `make_*` factory). Stage footholds under
/// `cuda/gpw/internal/` are composed inside `run()` and are never the user API.
///
/// Hybrid composition (stated, not silent):
///   * host — Γ STVnl and shared AO product / Bloch tables once per geometry;
///     multi-k payload diag / cold-start; Gamma single-grid sparse AO force
///     coefficients once per geometry; UKS/MG/multi-k forces and fixed-P stress
///   * device mid-SCF — SG: product pack→PBC collocate→KS→integrate→scatter;
///     MG: monom multi-level pack→`KsMultiGridChainDevice`→scatter (host collint
///     only for Γ with `ATV_GPW_CUDA_HOST_MG=1`); UKS: dual pack + `run_uks`;
///     multi-k: device STVnl S/T, Bloch AO contractions on a single grid or
///     phase contractions around the shared product multigrid chain, then
///     `KpointDiisEigenDevice`. Γ RKS/UKS: device H=H0+V,
///     `AoDiisEigenDevice` DIIS/eigen/density, final `eigen_from_device`
///     payload (no host re-diag). The method owns typed GPW stages; standalone
/// execution owns a Backend, nested execution borrows the parent stream.
/// Gamma RKS single-grid force contractions consume accepted device P,
/// physical final-H W and resident KS grids; scientific payload retrieval
/// follows the force body. Other force paths retain their host assembly.
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
///   auto sigma = m.stress(frame);  // eV/Å³; kStress = sigma * volume [eV]
/// @endcode
#pragma once

#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/driver/driver_node.h"
#include <cuda_runtime.h>
#include <memory>

#include <array>
#include <utility>
#include <vector>

namespace ATV {
namespace gaussian_core {
class Model;
}
namespace cuda {
namespace gpw {

/// @brief Result type shared verbatim with the CPU method (energy in Hartree).
using GpwResult = ::ATV::cpu::gpw::GpwResult;
struct GpwLoopState;
namespace internal {
struct GpwDftAccess;
class GpwForceWorkspace;
} // namespace internal

/// @brief Concrete GPW DFT method on CUDA — members + `build` / `run` /
///        `last_result`.
///
/// Product slice: Γ DIIS RKS (`n_occ`, PADE/PBE family, single grid or REL_CUTOFF
/// multigrid), collinear UKS (`n_alpha`/`n_beta`, PADE/PBE family, single grid), or
/// multi-k RKS DIIS (`k_mesh` / `k_points`, single grid or multigrid, fixed occ), periodic.
/// Move-only.
class GpwDft : public cuda::Driver<ATV::gpw::Phase>, public cuda::DriverNode {
  public:
    // ── Identity (constructor arguments, same as `cpu::gpw::GpwDft`) ─────
    /// @brief Per-element basis + GTH potential. `build(Frame)` / `run(Frame)`
    ///        map `kAtom_atomic_number` (Z) onto these entries.
    const std::vector<cpu::gpw::Kind> kinds;
    const double cutoff_ry; ///< density grid cutoff [Ry].
    /// @brief RKS/UKS: PADE or PBE/PBEsol/RevPBE/RPBE.
    const cpu::xc::XCType xc;

    // ── Knobs ──────────────────────────────────────────────────────────
    int ngrids = 1;          ///< multigrid levels (≥1; >1 needs rel_cutoff>0).
    double rel_cutoff = 0.0; ///< CP2K REL_CUTOFF [Ry] (required when ngrids>1).
    int n_occ = 0;           ///< RKS: doubly-occupied orbitals (N_elec / 2).
    int n_alpha = 0;         ///< UKS α occupations (with `n_beta`; 0 ⇒ RKS).
    int n_beta = 0;          ///< UKS β occupations (with `n_alpha`; 0 ⇒ RKS).
    int max_iter = 100;      ///< max SCF iterations.
    double tol = 1e-8;       ///< SCF energy convergence [Hartree].
    /// Maximum orthonormal KS-density commutator for DIIS [Hartree].
    /// Tighten alongside tol for precise derivative comparisons.
    double residual_tol = 1e-5;
    double eps_default = 1e-8; ///< Gaussian-tail cutoff.
    double mix = 0.5;          ///< linear density mixing when diis_depth==0.
    int diis_depth = 6;        ///< Pulay DIIS history depth.
    double lindep_threshold = 1e-8; ///< S-eigenvalue keep threshold.
    double level_shift = 0.0;       ///< virtual level shift [Hartree] (Γ path).
    bool nonperiodic = false;       ///< MT Poisson — not on device (throws).
    bool local_pp_on_grid = true;   ///< collocate GTH local SR PP on grid.
    bool use_ot = false;            ///< OT — not on device product (throws).
    /// @brief Monkhorst–Pack subdivisions (CP2K convention). `{1,1,1}` is Γ
    ///        product path; any \(N_d>1\) selects multi-k RKS DIIS.
    int k_mesh[3] = {1, 1, 1};
    /// @brief Explicit fractional k list (overrides `k_mesh` when non-empty).
    std::vector<std::array<double, 3>> k_points;
    /// @brief Weights for `k_points` (must match length; Σ = 1).
    std::vector<double> k_point_weights;
    /// @brief Time-reversal fold of the Monkhorst–Pack mesh (CPU twin default).
    bool k_time_reversal = true;
    /// @brief Dimensionless central strain step used by `stress()` (fixed P).
    double stress_strain_h = 2e-3;
    /// @brief Per-axis stress mask (x,y,z); masked entries report zero.
    std::array<bool, 3> stress_axes = {true, true, true};

    /// @brief GPW Kohn–Sham method (same signature as `cpu::gpw::GpwDft`).
    /// @param kinds One basis + GTH entry per element in the Frames to solve.
    /// @param cutoff_ry Density grid cutoff [Ry].
    /// @param xc PADE or PBE/PBEsol/RevPBE/RPBE for either spin mode.
    GpwDft(std::vector<cpu::gpw::Kind> kinds, double cutoff_ry,
           cpu::xc::XCType xc);
    ~GpwDft() override;
    GpwDft(const GpwDft&) = delete;
    GpwDft& operator=(const GpwDft&) = delete;
    /// Move an unattached method; a live Runtime binding rejects moves.
    GpwDft(GpwDft&&);
    GpwDft& operator=(GpwDft&&) = delete;

    /// @brief Validate against a Frame and copy its orthorhombic cell.
    ///
    /// Reads the cell from `frame.box` [Å → Bohr]. Each atom's Z must match one
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

    /// @brief DIIS SCF at the Frame geometry. Does not write the Frame.
    /// @param frame Geometry [Å] + Z. The cell is read from `frame.box`.
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

    /// @brief One device SCF followed by host fixed-P cell-stress assembly.
    ///
    /// Uses the converged result of this call without a second SCF. Central
    /// differences at `stress_strain_h` and `stress_axes` rebuild the host
    /// operators at fixed fractional coordinates and density matrix. This is
    /// a hybrid assembly, not a device-only or closed-form stress tensor.
    /// Only orthorhombic diagonal strains are implemented; shear is zero.
    /// Overwrites `kEnergy` [eV] and the extensive virial `kStress = sigma *
    /// volume` [eV], preserving the Frame's forces. Sign convention:
    /// sigma = -(1/volume) dE/dstrain, matching the CPU GPW method.
    /// @param frame Positions [Å], species Z, orthorhombic cell [Å].
    /// @return Intensive stress [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
    /// @throws std::logic_error when called before `build()`.
    /// @throws std::invalid_argument for unsupported settings or invalid strain.
    /// @throws std::runtime_error if the SCF does not converge.
    std::array<double, 6> stress(cpu::Frame& frame);

    /// @brief Result of the most recent `run()` (`converged==false` before first).
    const GpwResult& last_result() const { return result_; }

    /// Electronic source mounted directly on an outer CUDA Driver.
    bool provides_energy() const { return true; }
    bool provides_force() const { return true; }
    bool provides_system_stress() const { return true; }
    int default_budget() const override { return max_iter; }
    void build_on(cuda::Context& ctx, cuda::Frame& frame) override;
    int run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                          int budget) override;
    void destroy_on(cuda::Context& ctx, cuda::Frame& frame) override;

  protected:
    void validate_components() const override {}

  private:
    friend struct internal::GpwDftAccess;
    using GpwAtom = cpu::gpw::internal::GpwAtom;
    using GpwGeometry = cpu::gpw::internal::GpwGeometry;
    using GpwWorkspace = cpu::gpw::internal::GpwWorkspace;
    void validate_slice_() const;
    std::vector<GpwAtom> prepare_(const cpu::Frame& frame);
    GpwGeometry assemble_geometry_(std::vector<GpwAtom> atoms) const;
    void release_backend_();
    void drive_stages_(cuda::Context& ctx, cuda::Frame& frame, int budget);
    GpwResult&
    solve_(const GpwGeometry& geometry, GpwWorkspace& workspace,
           const cpu::linalg::RealMatrix* guess_C,
           const cpu::linalg::RealMatrix* guess_P,
           const cpu::linalg::RealMatrix* guess_P_beta, cuda::Context& ctx,
           cuda::Frame& frame, int budget,
           std::vector<std::array<double, 3>>* resident_force = nullptr,
           cuda::Frame* resident_target = nullptr,
           std::array<double, 6>* resident_stress = nullptr);
    GpwResult& solve_kpoints_(const GpwGeometry& geometry,
                              GpwWorkspace& workspace, cuda::Context& ctx,
                              cuda::Frame& frame, int budget);
    GpwResult& standalone_(const cpu::Frame& frame,
                           const cpu::linalg::RealMatrix* guess_C,
                           const cpu::linalg::RealMatrix* guess_P,
                           const cpu::linalg::RealMatrix* guess_P_beta,
                           std::vector<std::array<double, 3>>* forces,
                           std::array<double, 6>* stress);
    std::unique_ptr<internal::GpwForceWorkspace>
    prepare_force_workspace_(const GpwGeometry& geometry,
                             const gaussian_core::Model& core_overlap,
                             cudaStream_t stream);
    cpu::gpw::GpwDft cpu_shell_() const;
    std::vector<std::array<double, 3>>
    forces_from_state_(const GpwGeometry& geometry, const GpwResult& scf,
                       const GpwWorkspace& workspace, cudaStream_t stream,
                       cuda::Frame* accumulate_into = nullptr);
    std::array<double, 6>
    stress_from_state_(const GpwGeometry& geometry, const GpwResult& scf,
                       const GpwWorkspace& workspace) const;

    double cell_[3] = {0, 0, 0}; // Private per-call geometry, from Frame only.
    GpwResult result_{};
    bool built_ = false;
    std::unique_ptr<GpwLoopState> loop_;
};

} // namespace gpw
} // namespace cuda
} // namespace ATV

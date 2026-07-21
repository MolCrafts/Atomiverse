/// @file cpu/gpw/gpw_solver.h
/// @brief Self-contained GPW (Gaussian and Plane Waves) closed-shell SCF solver.
///
/// Structurally parallel to `cpu::scf::SCFSolver` and `cpu::pw::PWSolver`: a
/// self-contained inner solver for one electronic-structure method. The Kohn-Sham
/// problem is solved in the contracted Gaussian AO basis (small dense generalized
/// eigenproblem H C = S C ε), while the Hartree + XC + long-range local
/// pseudopotential are evaluated on a uniform real-space grid (collocate / FFT
/// Poisson / integrate). It reuses only shared *physics/algebra* from public
/// headers — `cpu::linalg` (generalized eigensolve, density build) and
/// `cpu::xc::pade` (functional) — never another domain's `internal`.
///
/// Scope: closed-shell, Gamma point, LDA (PADE). Periodic neutral electrostatics
/// via the GTH Gaussian core charge. Forces / GGA / nonlocal-for-all-l and full
/// CP2K bit-parity are follow-on milestones.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h" // GaussianBasis
#include "atomiverse/cpu/gpw/gth_potential.h" // GthPotential, GthNonlocalChannel
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/linalg/matrix.h" // RealMatrix

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief One atom's GPW data: valence charge, basis, and GTH pseudopotential.
struct GpwAtom {
    double z_val = 0.0;        ///< valence charge (electrons it contributes).
    int z = 0;                 ///< atomic number Z (for DFT-D3; 0 = unset).
    double pos[3] = {0, 0, 0}; ///< position [Bohr].
    std::vector<GridBasisFunction>
        basis;                ///< contracted GTOs, already centered at pos.
    double rloc = 0.0;        ///< GTH local radius.
    std::vector<double> cloc; ///< GTH local short-range C coefficients.
    std::vector<GthNonlocalChannel>
        nonlocal; ///< GTH nonlocal projector channels.

    GpwAtom() = default;

    /// @brief Construct from a contracted basis + GTH potential.
    ///
    /// Converts each contracted shell of `basis` into Cartesian
    /// GridBasisFunctions centered at `pos`, folding the per-primitive Gaussian
    /// normalization N(alpha,L) = (2 alpha/pi)^{3/4} (4 alpha)^{L/2} into the
    /// coefficient (the L alpha-power is what fixes the spanned space to match
    /// CP2K; the constant factor is irrelevant to the energy). Angular
    /// functions are the (2L+1) pure-spherical real solid harmonics from
    /// real_solid_harmonics(L), for any L.
    /// @param pos   Atom position [Bohr]; z_val is taken from `pot.zion`.
    /// @param basis Contracted Gaussian basis to center at `pos`.
    /// @param pot   GTH pseudopotential (local + nonlocal channels).
    /// @param z     Atomic number Z (needed only for DFT-D3, GpwConfig.d3).
    GpwAtom(const double pos[3], const GaussianBasis& basis,
            const GthPotential& pot, int z = 0);
};

/// @brief Exchange-correlation functional choice.
enum class XcKind { PADE, PBE };

/// @brief GPW SCF configuration.
struct GpwConfig {
    double cell[3] = {0, 0, 0}; ///< orthorhombic box lengths [Bohr].
    double cutoff_ry =
        300.0; ///< density grid cutoff [Ry] (finest multigrid level).
    int ngrids =
        1; ///< CP2K multigrid levels; 1 = single grid (default, bit-identical).
    double rel_cutoff =
        0.0;       ///< CP2K REL_CUTOFF [Ry]; exponent->level map when ngrids>1.
    int n_occ = 0; ///< number of doubly-occupied orbitals (N_elec / 2).
    int max_iter = 100; ///< max SCF iterations.
    double tol = 1e-8;  ///< SCF energy convergence [Hartree].
    double eps_default =
        1e-8; ///< CP2K EPS_DEFAULT: Gaussian-tail cutoff driving the
              ///< collocation window and the Γ-point lattice-image count.
    double mix =
        0.5; ///< linear density-mixing factor (used only when diis_depth == 0).
    int diis_depth =
        6; ///< Pulay DIIS history depth; 0 disables DIIS (falls back to linear `mix`).

    /// @brief Keep-threshold on the overlap eigenvalues for canonical
    /// orthogonalization (dimensionless). Directions with s_i < this are DROPPED
    /// rather than amplified by 1/sqrt(s_i). Diffuse MOLOPT functions in a dense
    /// periodic cell drive S toward linear dependence — without this the raw
    /// generalized solver fails outright (a 13726-AO slab aborted with
    /// `dsygvd failed, info=20590`) and the SCF oscillates. Matches
    /// cpu::scf::SCFConfig::lindep_threshold. 0 disables filtering.
    double lindep_threshold = 1e-8;

    /// @brief Virtual-orbital level shift [Hartree] applied inside the SCF's
    /// diagonalization (0 = off). Widens the effective HOMO-LUMO gap so a
    /// small-gap system stops swapping occupied/virtual states between iterations.
    /// It biases only the SCF PATH, not the fixed point: at convergence the
    /// occupied space stops changing and the converged density is the unshifted
    /// one. Needed for ZnO, which otherwise oscillates for 400 iterations.
    double level_shift = 0.0;
    XcKind xc = XcKind::PADE; ///< LDA (PADE) or GGA (PBE).
    bool nonperiodic = false; ///< use the isolated (MT) Poisson solver.
    /// @brief Collocate the GTH local short-range PP on the grid (CP2K's route,
    /// O(grid)) rather than building it analytically per basis pair
    /// (O(nb²·natoms)). Energy and analytic force BOTH read this flag: the
    /// gradient must be taken of the functional the SCF converged, or the two
    /// disagree by the grid discretization error of V_loc^SR (see
    /// internal/local_pp.h). Flip to false only to cross-check the two routes.
    bool local_pp_on_grid = true;
    bool d3 = false; ///< add Grimme DFT-D3 (zero damping, PBE) dispersion
                     ///< to the energy and forces (requires GpwAtom.z set);
                     ///< parameters come from the in-tree tables
                     ///< (basis_data/dftd3_tables.h).
    /// @brief Use the OT (orbital transformation) minimizer instead of dense
    /// diagonalization. Diagonalization-free SCF (gradient O(nb²·n_occ)); the
    /// path to large systems where the O(nb³) eigensolve is prohibitive.
    /// Converges to the same gapped ground state as the diagonalization path.
    bool use_ot = false;
};

/// @brief GPW SCF result with an energy decomposition.
struct GpwResult {
    double e_total = 0.0;    ///< total energy [Hartree].
    double e_kinetic = 0.0;  ///< Tr(P T).
    double e_loc_sr = 0.0;   ///< Tr(P V_loc^sr).
    double e_nonlocal = 0.0; ///< Tr(P V_nl).
    double e_es = 0.0; ///< total electrostatic (Hartree + e-ion-lr + ion-ion).
    double e_xc = 0.0; ///< XC energy.
    bool converged = false;
    int iters = 0;

    /// @name Converged wavefunction (populated on convergence; empty otherwise).
    /// Consistent set from a final diagonalization of the converged KS matrix:
    /// `P == build_density_matrix(C, n_occ)`. Used to assemble analytic forces
    /// (energy-weighted density matrix W = Σ_i 2 ε_i c_i c_iᵀ).
    /// @{
    linalg::RealMatrix C;    ///< MO coefficients, columns = orbitals (nb×nb).
    linalg::RealMatrix P;    ///< Converged density matrix (nb×nb).
    std::vector<double> eps; ///< Orbital energies ε_i [Hartree], length nb.
    /// @}
};

/// @brief Inner GPW solver — the `*Solver` face of the GPW method
///        (mirrors `cpu::pw::PWSolver` / `cpu::scf::SCFSolver`).
///
/// **NOT a `Driver<...>` subclass** — this is the inner solver behind
/// `GPWEvaluator`. Construction fixes the configuration; `build()` runs
/// the one-time fail-fast validation against a reference atom set (the
/// no-fallback rule: invalid configuration throws instead of degrading);
/// `run()` is re-entrant and rebuilds every geometry-dependent structure
/// per call (grids, lattice images, S/T/Vpp — same policy as `PWSolver`).
/// Move-only, like the sibling solvers.
class GPWSolver {
  public:
    /// @brief Fix the GPW configuration for this solver.
    /// @param config Grid/XC/electrostatics configuration (cell in [Bohr]).
    explicit GPWSolver(GpwConfig config);

    GPWSolver(const GPWSolver&) = delete;
    GPWSolver& operator=(const GPWSolver&) = delete;
    GPWSolver(GPWSolver&&) = default;
    GPWSolver& operator=(GPWSolver&&) = default;

    /// @brief One-time fail-fast validation against a reference system.
    ///
    /// @param reference Atoms the solver will be run on (used for
    ///        structural validation only; `run()` takes the live set).
    /// @throws std::invalid_argument when the atom set is empty, when
    ///         `config.n_occ < 1`, or when a periodic cell edge is
    ///         non-positive.
    void build(const std::vector<GpwAtom>& reference);

    /// @brief Run the closed-shell GPW SCF at the given geometry.
    ///
    /// Re-entrant: geometry-dependent structures are rebuilt per call.
    /// @param atoms Atom set at the current geometry [Bohr].
    /// @param guess_C Optional warm-start MO coefficients from an earlier run.
    /// @return The owned result of this run (valid until the next `run()`).
    /// @throws std::logic_error when called before `build()`.
    GpwResult& run(const std::vector<GpwAtom>& atoms,
                   const linalg::RealMatrix* guess_C = nullptr);

    /// @brief Result of the most recent `run()` (default-constructed —
    ///        `converged == false` — before the first run).
    const GpwResult& last_result() const { return result_; }

    /// @brief Configuration this solver was constructed with.
    const GpwConfig& config() const { return config_; }

  private:
    GpwConfig config_;   ///< Fixed at construction.
    GpwResult result_;   ///< Owned result buffer (last run).
    bool built_ = false; ///< Whether build() validated the configuration.
};

/// @brief Finite-difference forces F_a = -dE/dR_a via central differences.
///
/// Runs a `GPWSolver` at +/- h displacements of each atom along each axis
/// (6*N_atoms SCF solves). Correct but expensive; analytic GPW forces are a
/// follow-on. Returns forces [Hartree/Bohr], one xyz triple per atom.
/// @param h Displacement step [Bohr].
std::vector<std::array<double, 3>>
gpw_forces_fd(const std::vector<GpwAtom>& atoms, const GpwConfig& cfg,
              double h = 1e-3);

} // namespace gpw
} // namespace cpu
} // namespace ATV

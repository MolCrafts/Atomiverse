/// @file cpu/pw/pw_solver.h
/// @brief cpu::pw::PWSolver — public low-level inner plane-wave SCF solver.
///
/// **NOT a `Driver<...>` subclass** — `PWSolver` is the *inner* solver
/// (the plane-wave self-consistency loop) that lives behind the
/// `cpu::pw::PWEvaluator` Evaluator component, exactly mirroring the
/// gaussian-basis `cpu::scf::SCFSolver`. The user-facing way to compute
/// a plane-wave single-point energy is `cpu::Evaluate +
/// cpu::pw::PWEvaluator`; `PWSolver` is also a public API for low-level
/// use (debugging, eigenvalue dumps, convergence studies).
///
/// Its step helpers (`internal::PoissonSolver`, `XCBuilder`,
/// `VeffBuilder`, `DavidsonSolver`, `DensityBuilder`,
/// `ConvergenceCheck`) live in `cpu/pw/internal/`, are owned by this
/// class, never inherit `cpu::Component`, and are never registered
/// into any Plan.
///
/// Usage (low-level direct SCF):
/// @code
///   cpu::pw::PWConfig cfg;
///   cfg.ecut_ry = 30.0;
///   cfg.k_mesh = 4;
///   cpu::pw::PWSolver solver{cfg};
///   solver.build(frame);
///   const auto& result = solver.run(frame);
///   // result.energy [Ry], result.energy_ha [Ha], result.converged
/// @endcode
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>
// std::array used by pw_stress

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace pw {

// Facade-private step helpers and workspace — forward-declared only.
// Definitions live in include/atomiverse/cpu/pw/internal/ and are
// included from pw_solver.cpp; this public header must not include
// internal/*.h.
namespace internal {
class PoissonSolver;
class XCBuilder;
class VeffBuilder;
class DavidsonSolver;
class DensityBuilder;
class ConvergenceCheck;
} // namespace internal

/// @brief Plane-wave DFT configuration.
struct PWConfig {
    std::string upf_path;  ///< UPF v1 pseudopotential path (empty = toy V_loc).
    double ecut_ry = 30.0; ///< Wavefunction cutoff [Ry].
    int k_mesh = 4;        ///< Monkhorst-Pack mesh size per dimension.
    double tol = 1e-6;     ///< SCF tolerance on the density-change L2 norm.
    int max_iter = 50;     ///< Maximum SCF iterations.
    double mix_beta = 0.7; ///< Density mixing parameter (QE mixing_beta).
    int pulay_depth = 5;   ///< Pulay mixing history window.
};

/// @brief Result of a plane-wave DFT total energy calculation.
struct PWResult {
    double energy = 0;      ///< Total energy [Ry] (QE-comparable).
    double energy_ha = 0;   ///< Total energy [Hartree].
    bool converged = false; ///< Whether SCF converged within max_iter.
    int iterations = 0;     ///< Actual SCF iterations taken.
};

/// @brief Inner plane-wave SCF loop. Public low-level surface.
///
/// Construction → build(reference frame) → run(frame).
/// `last_result()` is valid after each `run()` call.
///
/// `build()` performs the geometry-independent setup (UPF parse).
/// Geometry-dependent structures (Ewald sums, G-vectors, FFT plans,
/// V_loc, KB projectors) are rebuilt inside each `run()` — atoms and
/// cell may change between runs (AIMD, optimization); grid-sized
/// scratch is reused when the FFT dimensions are unchanged.
///
/// When the project is built with MPI (`ATV_HAS_MPI`) and MPI is
/// initialized, `run()` distributes k-points round-robin across the
/// ranks of `MPI_COMM_WORLD` and Allreduces the band-energy and density
/// accumulations; every rank returns the identical converged result.
class PWSolver {
  public:
    /// @brief Construct from a plane-wave configuration.
    /// @param config Cutoff, k-mesh, convergence, mixing parameters.
    explicit PWSolver(PWConfig config);

    ~PWSolver();
    PWSolver(PWSolver&&) noexcept;
    PWSolver& operator=(PWSolver&&) noexcept;
    PWSolver(const PWSolver&) = delete;
    PWSolver& operator=(const PWSolver&) = delete;

    /// @brief One-time setup: parses the UPF pseudopotential (if any).
    /// @param reference Frame used only to validate atom types are
    ///                  supported; geometry is re-read at each `run()`.
    void build(const cpu::Frame& reference);

    /// @brief Execute the plane-wave SCF at the frame's current geometry.
    /// @param frame Atom positions [Å], types (Z), and simulation cell.
    /// @return Reference to the owned `PWResult` (valid until the next
    ///         `run()` or destruction).
    PWResult& run(const cpu::Frame& frame);

    /// @brief Access the most recent SCF result.
    const PWResult& last_result() const { return result_; }

    /// @brief Access the configuration (for inspection).
    const PWConfig& config() const { return config_; }

  private:
    PWConfig config_;
    bool built_ = false;

    // Step helpers — held by pointer so this public header only needs
    // the forward declarations above. The per-run problem (workspace,
    // G-spheres, k-mesh, projector tables) is assembled inside run()
    // by the shared internal::setup_pw_problem().
    std::unique_ptr<internal::PoissonSolver> poisson_;
    std::unique_ptr<internal::XCBuilder> xc_;
    std::unique_ptr<internal::VeffBuilder> veff_;
    std::unique_ptr<internal::DensityBuilder> density_;
    std::unique_ptr<internal::DavidsonSolver> davidson_;
    int davidson_bands_ = 0; ///< n_bands the current davidson_ was built for.

    // Owned result buffer (returned by reference from run()).
    PWResult result_;
};

/// @brief Finite-difference ionic forces F_a = −dE/dR_a via central differences.
///
/// Runs a full plane-wave SCF at ±h displacements of each atom along each
/// Cartesian axis (6 N_atoms SCF solves). Retained as the numerical oracle
/// for analytic forces (`pw_forces_analytic`).
///
/// @param frame Geometry [Å]; not mutated.
/// @param cfg   Plane-wave configuration (UPF path, ecut, k-mesh, SCF tol).
/// @param h_ang Displacement step [Å] (default 1e-3).
/// @return Forces [eV/Å], one xyz triple per atom, same order as frame rows.
/// @throws std::invalid_argument if h_ang ≤ 0 or the frame has no atoms.
std::vector<std::array<double, 3>>
pw_forces_fd(const cpu::Frame& frame, const PWConfig& cfg, double h_ang = 1e-3);

/// @brief Analytic ionic forces (local HF + Ewald + KB nonlocal).
///
/// Default force engine for `PWEvaluator::with_gradient(true)`. One SCF plus
/// a gradient pass (local/Ewald from ρ; nonlocal from a final diagonalization
/// per k-point). FD oracle remains `pw_forces_fd`.
///
/// @param frame Geometry [Å]; not mutated.
/// @param cfg   Plane-wave configuration.
/// @return Forces [eV/Å], one xyz triple per atom.
/// @throws std::invalid_argument if the frame has no atoms.
/// Non-converged SCF: forces are still returned from the last density (with
/// a warning log), matching the practical FD oracle behaviour.
std::vector<std::array<double, 3>> pw_forces_analytic(const cpu::Frame& frame,
                                                      const PWConfig& cfg);

/// @brief Cell stress σ [eV/Å³], Voigt order xx,yy,zz,xy,xz,yz.
///
/// v1: central FD of the SCF energy under symmetric cell strain at fixed
/// fractional coordinates (same oracle class as early ionic FD forces).
/// @param frame Geometry [Å]; not mutated.
/// @param cfg   Plane-wave configuration.
/// @return Stress [eV/Å³] written into Frame as `kStress` by the evaluator.
std::array<double, 6> pw_stress(const cpu::Frame& frame, const PWConfig& cfg);

} // namespace pw
} // namespace cpu
} // namespace ATV

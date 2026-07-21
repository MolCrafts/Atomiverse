/// @file cuda/scf/scf_solver.cuh
/// @brief cuda::scf::SCFSolver — public low-level inner SCF solver.
///
/// **NOT a `Driver<...>` subclass** — `SCFSolver` is the *inner* solver
/// (the SCF self-consistency loop) that lives behind force-providing
/// evaluator components. The user-facing way to compute single-point
/// CUDA electronic structure is `cuda::Evaluate + cuda::scf::SCF`;
/// `SCFSolver` is the internal machinery `SCF` calls.
///
/// `SCFSolver` is also a public API for low-level use (debugging, testing,
/// CPU/CUDA parity). It owns its `internal::*` step helpers, each with a
/// `build` / `step` / `destroy` lifecycle and pre-allocated device
/// workspaces. The helpers are **not** registered into any `Driver` Plan,
/// carry no phase enum, and cannot touch any component-layer state — the
/// solver's entire external surface is `(void* stream, cuda::Frame&)`,
/// guaranteed by the type system (and the `test_cuda_scf_solver_isolation`
/// grep gate) rather than by comment-level promise; `SCFSolver` drives
/// them directly in a fixed order and owns its own internal convergence.
///
/// Three run paths are surfaced as distinct constructors:
///  - **Synthetic** — `H(P) = H0 + alpha*P` toy self-consistency cycle.
///  - **RHF (host integrals)** — Fock build from a precomputed
///    `cpu::scf::IntegralPack` (`ATV_HAS_HF` only).
///  - **RHF (device integrals)** — Fock build from GPU-native integral
///    kernels, zero CPU integral computation (`ATV_HAS_HF` only).
///
/// Usage (low-level direct SCF):
/// @code
///   cuda::scf::SCFSolver solver{n_occ, alpha, tol, max_iter};
///   solver.build(stream, frame);   // stream: void* (cudaStream_t)
///   solver.run(stream, frame);
///   // solver.workspace().converged / .residual and frame.scalars->energy
///   // are populated.
/// @endcode
///
/// Usage (single-point through the outer facade):
/// @code
///   cuda::Evaluate eval;
///   eval.on(evaluate::Phase::evaluate).add(cuda::scf::SCF{...});  // owns an SCFSolver internally
///   eval.build();
///   eval.run(frame);
/// @endcode
#pragma once

#include <memory>
#include <vector>

#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "atomiverse/cuda/scf/internal/density_builder.cuh"
#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"
#include "atomiverse/cuda/scf/internal/hamiltonian.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

#ifdef ATV_HAS_HF
#include "atomiverse/cuda/scf/internal/rhf_density.cuh"
#include "atomiverse/cuda/scf/internal/rhf_fock.cuh"
namespace ATV {
namespace cpu {
namespace scf {
struct IntegralPack; // cpu/scf/internal/integrals.h — not a public include
} // namespace scf
} // namespace cpu
} // namespace ATV
#endif

namespace ATV {
namespace cuda {
namespace scf {

/// @brief Inner CUDA SCF self-consistency loop. Public low-level surface.
///
/// Construction → `build(runtime, frame)` (one-time device-workspace
/// allocation) → `run(runtime, frame)` which loops the SCF cycle
/// internally to convergence or a max-iteration cap. The density matrix
/// is zeroed internally as the first step of `run()`.
class SCFSolver {
  public:
    /// @brief Construct a synthetic-path solver `H(P) = H0 + alpha*P`.
    /// @param n_occ Number of occupied orbitals [dimensionless].
    /// @param alpha Density coupling parameter [dimensionless].
    /// @param tol Convergence tolerance, Frobenius norm of P change
    ///        [dimensionless].
    /// @param max_iter Maximum number of SCF iterations [dimensionless].
    SCFSolver(int n_occ, double alpha, double tol, int max_iter);

#ifdef ATV_HAS_HF
    /// @brief Construct an RHF solver driven by host-computed integrals.
    /// @param integrals Precomputed molecular integral pack
    ///        (Hcore, ERI, S, V_nuc).
    /// @param tol Convergence tolerance, Frobenius norm of P change
    ///        [dimensionless].
    /// @param max_iter Maximum number of SCF iterations [dimensionless].
    /// @param level_shift Virtual-orbital level shift λ [Hartree]
    ///        (Saunders–Hillier), default 0.0 = no shift. Stabilizes SCF
    ///        by widening the HOMO–LUMO gap; reported orbital energies are
    ///        de-shifted after convergence. Mirrors CPU
    ///        SCFConfig::level_shift (hf-robust-03).
    /// @throws std::invalid_argument when `integrals.n_electrons` is odd or
    ///         negative: closed-shell RHF requires an even electron count,
    ///         and an odd count would silently truncate
    ///         (n_occ = n_electrons / 2 drops an electron). Mirrors the CPU
    ///         `cpu::scf::SCFSolver::run` occupation guard (hf-robust-01).
    SCFSolver(const cpu::scf::IntegralPack& integrals, double tol, int max_iter,
              double level_shift = 0.0);

    /// @brief Construct an RHF solver driven by GPU-native integral kernels.
    /// @param n_prim Primitives per shell (3 for STO-3G) [dimensionless].
    /// @param exponents Gaussian exponents [n_prim] [Bohr^-2].
    /// @param coefficients Raw contraction coefficients [n_prim]
    ///        (NOT normalized).
    /// @param n_occ Number of doubly-occupied orbitals [dimensionless].
    /// @param tol Convergence tolerance, Frobenius norm of P change
    ///        [dimensionless].
    /// @param max_iter Maximum number of SCF iterations [dimensionless].
    /// @throws std::invalid_argument when `n_occ` is negative.
    SCFSolver(int n_prim, const double* exponents, const double* coefficients,
              int n_occ, double tol, int max_iter);
#endif

    /// @brief One-time setup: allocate device workspaces and build the
    ///        owned `internal::*` step helpers.
    /// @param stream CUDA stream (`cudaStream_t` passed as `void*` to keep
    ///        the header CUDA-free; the component wrapper forwards
    ///        `Context::stream`).
    /// @param frame Device-resident system state, sized at this call.
    void build(void* stream, cuda::Frame& frame);

    /// @brief Execute the SCF self-consistency cycle on `frame`.
    ///
    /// Zeroes the density matrix internally, then loops the fixed-order
    /// SCF cycle (Fock/Hamiltonian build → diagonalize → density build →
    /// convergence check) up to `max_iter` iterations, breaking early when
    /// the owned workspace's `converged` flag is set. Performs no device
    /// allocation.
    /// @param stream CUDA stream (`cudaStream_t` as `void*`).
    /// @param frame Device-resident system state, updated in place.
    void run(void* stream, cuda::Frame& frame);

    /// @brief Release the owned SCF workspace plus the cuBLAS/cuSOLVER
    ///        handles and per-helper scratch buffers held by the
    ///        `internal::*` step helpers.
    /// @param stream CUDA stream (`cudaStream_t` as `void*`).
    /// @param frame Device-resident system state.
    void destroy(void* stream, cuda::Frame& frame);

    /// @brief Number of SCF iterations executed by the most recent `run()`.
    /// @return Iteration count [dimensionless].
    int iterations() const { return iterations_; }

    /// @brief Access the owned SCF workspace (density matrix, MO
    ///        coefficients, orbital energies, convergence state).
    ///
    /// Valid after `build()`. Exposed so force-evaluator components
    /// (`SCF`) can forward SCF results to downstream observers
    /// and recorders.
    /// @return The device-resident SCF workspace.
    const SCFWorkspace& workspace() const { return workspace_; }

  private:
    /// @brief Run-path selector — fixes which `internal::*` helpers drive
    ///        the Fock/Hamiltonian build and density rebuild.
    enum class Path {
        Synthetic,          ///< H(P) = H0 + alpha*P toy cycle.
        RHFHostIntegrals,   ///< RHF Fock build from a host IntegralPack.
        RHFDeviceIntegrals, ///< RHF Fock build from GPU integral kernels.
    };

    Path path_;                ///< Selected run path.
    double tol_;               ///< Convergence tolerance (Frobenius norm).
    int max_iter_;             ///< Maximum SCF iteration count.
    double level_shift_ = 0.0; ///< Virtual level shift λ [Hartree] (RHF host).
    bool built_ = false;       ///< Whether build() has run.
    int iterations_ = 0;       ///< Iterations executed by the last run().

    /// @brief Device-resident SCF workspace owned by this solver.
    ///
    /// Relocated out of `cuda::Frame` (Frame purity rule); allocated by
    /// the `internal::*` helpers in `build()`, freed in `destroy()`.
    SCFWorkspace workspace_;

    // Step helpers shared by every path.
    std::unique_ptr<internal::EigenSolver> eigen_;
    std::unique_ptr<internal::ConvergenceCheck> convergence_;

    // Synthetic-path helpers.
    std::unique_ptr<internal::Hamiltonian> hamiltonian_;
    std::unique_ptr<internal::DensityBuilder> density_;

#ifdef ATV_HAS_HF
    // RHF-path helpers (one Fock builder is active per solver).
    std::unique_ptr<internal::RHFFock> rhf_fock_;
    std::unique_ptr<internal::RHFFockDevice> rhf_fock_device_;
    std::unique_ptr<internal::RHFDensity> rhf_density_;
#endif
};

} // namespace scf
} // namespace cuda
} // namespace ATV

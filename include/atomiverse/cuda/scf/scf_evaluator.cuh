/// @file cuda/scf/scf_evaluator.cuh
/// @brief cuda::scf::SCF — CUDA Evaluator component wrapping an
///        inner `SCFSolver`.
///
/// `SCF` is a real `cuda::Component` with
/// `role() == ComponentRole::Evaluator` and
/// `capabilities() == Energy | DensityMatrix | Orbitals`. It owns one
/// `cuda::scf::SCFSolver`, builds it once in `build()`, and runs the full
/// SCF self-consistency cycle in `launch()`, writing the converged energy
/// and density into the device frame.
///
/// The user-facing single-point CUDA SCF workflow is
/// `cuda::Evaluate + cuda::scf::SCF` (or the `cuda::single_point`
/// helper). Mirrors `cpu::scf::SCF` minus the CPU-only
/// `with_gradient` toggle and ASPC `predictor()` warm-start — there is no
/// CUDA SCF gradient kernel, so this evaluator is energy/density only.
///
/// @code{.cpp}
///   cuda::Evaluate eval;
///   eval.on(evaluate::Phase::evaluate).add(cuda::scf::SCF{integrals, tol, max_iter});
///   eval.build();
///   eval.run(frame);            // frame.scalars.energy populated
/// @endcode
///
/// @see cuda::scf::SCFSolver for the inner SCF self-consistency loop
/// @see cuda::Evaluate for the one-pass driver this component plugs into
#pragma once

#include <memory>
#include <string>

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/scf/scf_solver.cuh"
#include "atomiverse/facade/component_role.h"

#ifdef ATV_HAS_HF
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

/// @brief CUDA SCF energy/density evaluator component.
///
/// Owns an `SCFSolver`, built once in `build()` and run in `launch()`.
/// Three constructors map onto the three `SCFSolver` run paths. Declares no
/// default bindings — attach explicitly with `.on(phase).add()`.
class SCF : public cuda::Component {
  public:
    /// @brief Construct a synthetic-path SCF evaluator `H(P) = H0 + alpha*P`.
    /// @param n_occ Number of occupied orbitals [dimensionless].
    /// @param alpha Density coupling parameter [dimensionless].
    /// @param tol Convergence tolerance, Frobenius norm of P change
    ///        [dimensionless].
    /// @param max_iter Maximum number of SCF iterations [dimensionless].
    SCF(int n_occ, double alpha, double tol, int max_iter);

#ifdef ATV_HAS_HF
    /// @brief Construct an RHF evaluator driven by host-computed integrals.
    /// @param integrals Precomputed molecular integral pack
    ///        (Hcore, ERI, S, V_nuc).
    /// @param tol Convergence tolerance, Frobenius norm of P change
    ///        [dimensionless].
    /// @param max_iter Maximum number of SCF iterations [dimensionless].
    /// @param level_shift Virtual-orbital level shift λ [Hartree], default
    ///        0.0 (no shift). Forwarded to the inner SCFSolver
    ///        (hf-robust-03).
    SCF(const cpu::scf::IntegralPack& integrals, double tol, int max_iter,
        double level_shift = 0.0);

    /// @brief Construct an RHF evaluator driven by GPU-native integrals.
    /// @param n_prim Primitives per shell (3 for STO-3G) [dimensionless].
    /// @param exponents Gaussian exponents [n_prim] [Bohr^-2].
    /// @param coefficients Raw contraction coefficients [n_prim]
    ///        (NOT normalized).
    /// @param n_occ Number of doubly-occupied orbitals [dimensionless].
    /// @param tol Convergence tolerance, Frobenius norm of P change
    ///        [dimensionless].
    /// @param max_iter Maximum number of SCF iterations [dimensionless].
    SCF(int n_prim, const double* exponents, const double* coefficients,
        int n_occ, double tol, int max_iter);
#endif

    /// @brief Component name.
    /// @return "scf_evaluator".
    std::string name() const override { return "scf_evaluator"; }
    Role role() const override { return 0; }

    /// @brief Structural role of this component.
    /// @return ComponentRole::Evaluator.
    /* role via Potential/kind base */

    /// @brief Frame fields this component writes each launch.
    /// @return `Energy | DensityMatrix | Orbitals` — no `Force` (no CUDA
    ///         SCF gradient kernel exists).

    // Deliberately no default_bindings(): like the CPU SCF, this
    // evaluator is reusable across workflows (Evaluate, MD, Minimize), so
    // the target phase must be chosen explicitly with .on(phase).add() — a
    // hardcoded evaluate::Phase would misbind when added to another driver.

    /// @brief Build the owned `SCFSolver` (one-time device allocation).
    /// @param ctx Build context (CUDA stream + device frame).
    void build(const BuildContext& ctx) override;

    /// @brief Run the full SCF cycle, writing energy/density to the frame.
    /// @param ctx Launch context (CUDA stream + device frame + step index).
    void launch(const LaunchContext& ctx) override;

    /// @brief Release the inner `SCFSolver`'s device resources.
    /// @param ctx Destroy context (CUDA stream + device frame).
    void destroy(const DestroyContext& ctx) override;

    /// @brief Access the converged SCF workspace (density matrix, MO
    ///        coefficients, orbital energies, convergence state).
    ///
    /// Valid after `build()`. Downstream `finalize`-phase observers and
    /// recorders (`MullikenObs`, `DensityRecorder`) hold a reference to
    /// this evaluator and read the converged density through here, since
    /// the SCF workspace is no longer carried on `cuda::Frame`.
    /// @return The device-resident SCF workspace.
    const SCFWorkspace& workspace() const { return solver_->workspace(); }

  private:
    std::unique_ptr<SCFSolver> solver_; ///< Owned inner SCF solver.
};

} // namespace scf
} // namespace cuda
} // namespace ATV

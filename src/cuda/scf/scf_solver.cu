/// @file cuda/scf/scf_solver.cu
/// @brief cuda::scf::SCFSolver implementation — the inner SCF
///        self-consistency loop.
///
/// `SCFSolver` orchestrates its owned `internal::*` step helpers through
/// the fixed-order SCF cycle. It owns the `SCFWorkspace` (no longer part
/// of `cuda::Frame`) and passes it by reference into every helper call.
/// It absorbs the former `ZeroDensity` infrastructure component: the
/// density matrix is zeroed via `cudaMemsetAsync` as the first step of
/// `run()`. The solver is not a `Driver`, carries no phase enum, and can
/// no longer see any component-layer state — its whole external surface
/// is `(void* stream, cuda::Frame&)`, so the old comment-level isolation
/// promise is now enforced by the type system. Convergence is managed
/// internally via the workspace's `converged` flag.

#include "atomiverse/cuda/scf/scf_solver.cuh"

#include <cmath>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace scf {

// ── Kernel: de-shift the virtual orbital energies ────────────────────
// The level shift raised every virtual eigenvalue by exactly λ and left
// the occupied ones untouched; subtract λ back from eig[n_occ .. n-1] so
// the reported orbital energies are shift-free (hf-robust-03 de-shift).
__global__ void deshift_virtuals_kernel(prec_t* __restrict__ eig, prec_t lambda,
                                        int n_occ, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + n_occ;
    if (i < n)
        eig[i] -= lambda;
}

SCFSolver::SCFSolver(int n_occ, double alpha, double tol, int max_iter)
    : path_(Path::Synthetic), tol_(tol), max_iter_(max_iter) {
    eigen_ = std::make_unique<internal::EigenSolver>();
    // Synthetic path: ConvergenceCheck owns the energy (E = trace(H*P)).
    convergence_ = std::make_unique<internal::ConvergenceCheck>(tol, true);
    hamiltonian_ = std::make_unique<internal::Hamiltonian>(n_occ, alpha);
    density_ = std::make_unique<internal::DensityBuilder>();
}

#ifdef ATV_HAS_HF

SCFSolver::SCFSolver(const cpu::scf::IntegralPack& integrals, double tol,
                     int max_iter, double level_shift)
    : path_(Path::RHFHostIntegrals), tol_(tol), max_iter_(max_iter),
      level_shift_(level_shift) {
    // Closed-shell validity guard (mirrors the CPU cpu::scf::SCFSolver::run
    // occupation guard, hf-robust-01). The CUDA solver is RHF closed-shell
    // only — n_occ = integrals.n_occ() = n_electrons / 2. An odd electron
    // count silently truncates (2 * n_occ = n_electrons - 1), describing one
    // fewer electron; reject it at construction rather than converge to a
    // physically wrong state. One descriptive throw per the repo
    // "no fallback values" rule.
    if (integrals.n_electrons < 0 || integrals.n_electrons % 2 != 0) {
        throw std::invalid_argument(
            "cuda::scf::SCFSolver: closed-shell RHF requires an even, "
            "non-negative electron count (n_electrons="
            + std::to_string(integrals.n_electrons)
            + "); an odd count would truncate to "
            + std::to_string(integrals.n_occ())
            + " doubly-occupied orbitals and drop an electron. Check the "
              "molecular charge, or use the CPU unrestricted path for "
              "open-shell systems.");
    }
    eigen_ = std::make_unique<internal::EigenSolver>();
    // RHF path: RHFDensity owns the energy; the checker is residual-only.
    convergence_ = std::make_unique<internal::ConvergenceCheck>(tol, false);
    rhf_fock_ = std::make_unique<internal::RHFFock>(integrals, level_shift_);
    rhf_density_ = std::make_unique<internal::RHFDensity>(integrals.n_occ());
}

SCFSolver::SCFSolver(int n_prim, const double* exponents,
                     const double* coefficients, int n_occ, double tol,
                     int max_iter)
    : path_(Path::RHFDeviceIntegrals), tol_(tol), max_iter_(max_iter) {
    // Device-integrals path takes n_occ directly (no electron count to
    // cross-check), so the only meaningful guard is non-negativity — a
    // negative occupied count is nonsense (mirrors the CPU n_occ >= 0 arm).
    if (n_occ < 0) {
        throw std::invalid_argument(
            "cuda::scf::SCFSolver: n_occ must be non-negative (got "
            + std::to_string(n_occ) + ").");
    }
    eigen_ = std::make_unique<internal::EigenSolver>();
    convergence_ = std::make_unique<internal::ConvergenceCheck>(tol, false);
    rhf_fock_device_ = std::make_unique<internal::RHFFockDevice>(
        n_prim, exponents, coefficients, n_occ);
    rhf_density_ = std::make_unique<internal::RHFDensity>(n_occ);
}

#endif // ATV_HAS_HF

void SCFSolver::build(void* stream, cuda::Frame& frame) {
    internal::SolverContext ctx{stream, frame, 0};

    // The Fock/Hamiltonian builder allocates the SCF workspace (H, P, ...)
    // and must build first; the remaining helpers size off workspace_.n.
    switch (path_) {
    case Path::Synthetic:
        hamiltonian_->build(ctx, workspace_);
        eigen_->build(ctx, workspace_);
        density_->build(ctx, workspace_);
        convergence_->build(ctx, workspace_);
        break;
#ifdef ATV_HAS_HF
    case Path::RHFHostIntegrals:
        rhf_fock_->build(ctx, workspace_);
        eigen_->build(ctx, workspace_);
        rhf_density_->build(ctx, workspace_);
        convergence_->build(ctx, workspace_);
        break;
    case Path::RHFDeviceIntegrals:
        rhf_fock_device_->build(ctx, workspace_);
        eigen_->build(ctx, workspace_);
        rhf_density_->build(ctx, workspace_);
        convergence_->build(ctx, workspace_);
        break;
#endif
    }
    built_ = true;
}

void SCFSolver::run(void* stream, cuda::Frame& frame) {
    if (!built_)
        build(stream, frame);

    auto cu_stream = static_cast<cudaStream_t>(stream);
    auto& scf = workspace_;

    // ── Absorbed ZeroDensity: zero P before the first iteration ───────
    scf.converged = false;
    scf.iteration = 0;
    if (scf.n > 0)
        cudaMemsetAsync(scf.P, 0, scf.n * scf.n * sizeof(cuda::prec_t),
                        cu_stream);

    // ── Internal SCF self-consistency loop ────────────────────────────
    // Fixed-order cycle: build Fock/Hamiltonian → diagonalize →
    // build density → convergence check. Helpers no-op once
    // workspace_.converged is set; the loop also breaks early on it.
    iterations_ = 0;
    for (int iter = 0; iter < max_iter_; ++iter) {
        internal::SolverContext ctx{stream, frame, /*step=*/iter};

        switch (path_) {
        case Path::Synthetic:
            hamiltonian_->step(ctx, workspace_);
            eigen_->step(ctx, workspace_);
            density_->step(ctx, workspace_);
            convergence_->step(ctx, workspace_);
            break;
#ifdef ATV_HAS_HF
        case Path::RHFHostIntegrals:
            rhf_fock_->step(ctx, workspace_);
            eigen_->step(ctx, workspace_);
            rhf_density_->step(ctx, workspace_);
            convergence_->step(ctx, workspace_);
            break;
        case Path::RHFDeviceIntegrals:
            rhf_fock_device_->step(ctx, workspace_);
            eigen_->step(ctx, workspace_);
            rhf_density_->step(ctx, workspace_);
            convergence_->step(ctx, workspace_);
            break;
#endif
        }

        ++iterations_;
        if (scf.converged)
            break;
    }

    // ── Post-convergence de-shift (hf-robust-03) ──────────────────────
    // Reported orbital energies must not carry the +λ virtual pollution.
    // At convergence the shift's spectral effect is exactly +λ on every
    // virtual eigenvalue and 0 on occupied, so subtracting λ from the
    // virtual block restores unshifted energies without a re-diagonalize.
    if (scf.converged && std::abs(level_shift_) > 1e-4 && scf.n_occ < scf.n) {
        const int n_virt = scf.n - scf.n_occ;
        int blocks = cuda::grid_size(n_virt, cuda::kDefaultBlock);
        deshift_virtuals_kernel<<<blocks, cuda::kDefaultBlock, 0, cu_stream>>>(
            scf.eig, static_cast<cuda::prec_t>(level_shift_), scf.n_occ, scf.n);
    }
}

void SCFSolver::destroy(void* stream, cuda::Frame& frame) {
    internal::SolverContext ctx{stream, frame, 0};

    switch (path_) {
    case Path::Synthetic:
        convergence_->destroy(ctx, workspace_);
        density_->destroy(ctx, workspace_);
        eigen_->destroy(ctx, workspace_);
        hamiltonian_->destroy(ctx, workspace_);
        break;
#ifdef ATV_HAS_HF
    case Path::RHFHostIntegrals:
        convergence_->destroy(ctx, workspace_);
        rhf_density_->destroy(ctx, workspace_);
        eigen_->destroy(ctx, workspace_);
        rhf_fock_->destroy(ctx, workspace_);
        break;
    case Path::RHFDeviceIntegrals:
        convergence_->destroy(ctx, workspace_);
        rhf_density_->destroy(ctx, workspace_);
        eigen_->destroy(ctx, workspace_);
        rhf_fock_device_->destroy(ctx, workspace_);
        break;
#endif
    }

    // Free the device-resident SCF workspace arrays owned by this solver.
    // (Formerly freed by cuda::destroy(frame) when the workspace lived on
    // the frame; the helpers above release only their private scratch.)
    cudaFree(workspace_.H0);
    cudaFree(workspace_.H);
    cudaFree(workspace_.P);
    cudaFree(workspace_.P_old);
    cudaFree(workspace_.C);
    cudaFree(workspace_.eig);
    cudaFree(workspace_.work);
    cudaFree(workspace_.Hcore);
    cudaFree(workspace_.X);
    cudaFree(workspace_.F_prime);
    cudaFree(workspace_.ERI);
    cudaFree(workspace_.d_shell_alpha);
    cudaFree(workspace_.d_shell_coeff);
    cudaFree(workspace_.d_shell_center);
    cudaFree(workspace_.d_atom_Z);
    cudaFree(workspace_.d_atom_pos);
    workspace_ = {};

    built_ = false;
}

} // namespace scf
} // namespace cuda
} // namespace ATV

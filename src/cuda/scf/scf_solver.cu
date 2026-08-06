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

#include "atomiverse/cuda/scf/internal/scf_solver.cuh"

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

SCFSolver::SCFSolver(xt::xtensor<double, 2> S, xt::xtensor<double, 2> Hcore,
                     xt::xtensor<double, 4> ERI, double V_nuc, int n_electrons,
                     double tol, int max_iter, double level_shift)
    : path_(Path::RHFHostIntegrals), tol_(tol), max_iter_(max_iter),
      level_shift_(level_shift) {
    // Closed-shell RHF only: even non-negative electron count.
    if (n_electrons < 0 || n_electrons % 2 != 0) {
        throw std::invalid_argument(
            "cuda::scf::SCFSolver: closed-shell RHF requires an even, "
            "non-negative electron count (n_electrons="
            + std::to_string(n_electrons)
            + "). Check the molecular charge, or use the CPU unrestricted "
              "path for open-shell systems.");
    }
    const int n_occ = n_electrons / 2;
    eigen_ = std::make_unique<internal::EigenSolver>();
    convergence_ = std::make_unique<internal::ConvergenceCheck>(tol, false);
    rhf_fock_ = std::make_unique<internal::RHFFock>(
        std::move(S), std::move(Hcore), std::move(ERI), V_nuc, n_occ,
        level_shift_);
    rhf_density_ = std::make_unique<internal::RHFDensity>(n_occ);
}

SCFSolver::SCFSolver(int n_prim, const double* exponents,
                     const double* coefficients, int n_occ, double tol,
                     int max_iter, double level_shift)
    : SCFSolver(/*n_shells_per_atom=*/1, n_prim, exponents, coefficients, n_occ,
                tol, max_iter, level_shift) {}

SCFSolver::SCFSolver(int n_shells_per_atom, int n_prim,
                     const double* alpha_atom, const double* coeff_atom,
                     int n_occ, double tol, int max_iter, double level_shift)
    : path_(Path::RHFDeviceIntegrals), tol_(tol), max_iter_(max_iter),
      level_shift_(level_shift) {
    if (n_occ < 0) {
        throw std::invalid_argument(
            "cuda::scf::SCFSolver: n_occ must be non-negative (got "
            + std::to_string(n_occ) + ").");
    }
    if (n_shells_per_atom < 1 || n_prim < 1) {
        throw std::invalid_argument(
            "cuda::scf::SCFSolver: n_shells_per_atom and n_prim must be ≥ 1");
    }
    eigen_ = std::make_unique<internal::EigenSolver>();
    convergence_ = std::make_unique<internal::ConvergenceCheck>(tol, false);
    rhf_fock_device_ = std::make_unique<internal::RHFFockDevice>(
        n_shells_per_atom, n_prim, alpha_atom, coeff_atom, n_occ, level_shift_);
    rhf_density_ = std::make_unique<internal::RHFDensity>(n_occ);
}

SCFSolver::SCFSolver(int n_shells, int n_ao, int n_prim, const double* alpha_mol,
                     const double* coeff_mol, const int* L_mol,
                     const int* atom_of_mol, int n_occ, double tol,
                     int max_iter, double level_shift)
    : path_(Path::RHFDeviceIntegrals), tol_(tol), max_iter_(max_iter),
      level_shift_(level_shift) {
    if (n_occ < 0) {
        throw std::invalid_argument(
            "cuda::scf::SCFSolver: n_occ must be non-negative (got "
            + std::to_string(n_occ) + ").");
    }
    if (n_shells < 1 || n_ao < 1 || n_prim < 1) {
        throw std::invalid_argument(
            "cuda::scf::SCFSolver: n_shells, n_ao, n_prim must be ≥ 1");
    }
    eigen_ = std::make_unique<internal::EigenSolver>();
    convergence_ = std::make_unique<internal::ConvergenceCheck>(tol, false);
    rhf_fock_device_ = std::make_unique<internal::RHFFockDevice>(
        n_shells, n_ao, n_prim, alpha_mol, coeff_mol, L_mol, atom_of_mol,
        n_occ, level_shift_);
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
    // Fixed-order cycle: Fock → diagonalize → density → convergence.
    //
    // Device-integral path: residual / energy / V_nuc stay on device —
    // **no D2H during the loop**. Host `scf.converged` is filled once after
    // the last iteration (post-pass). Host-integral / synthetic paths still
    // early-exit via host residual (legacy).
    iterations_ = 0;
    const bool device_no_d2h =
#ifdef ATV_HAS_HF
        (path_ == Path::RHFDeviceIntegrals);
#else
        false;
#endif
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
        if (device_no_d2h) {
            // Mapped d_converged is written on-device by ConvergenceCheck.
            // Must poll every iteration: a coarser stride left the loop running
            // after residual passed, and build_G's sticky early-out then rebuilt
            // F = Hcore (G zeroed), overwriting the correct energy with the core
            // Hamiltonian energy (H₂: −1.12 Ha → −1.79 Ha oscillation).
            // One stream fence per iter is noise vs Fock/ERI for real molecules.
            cudaStreamSynchronize(cu_stream);
            if (scf.h_converged_mapped && *scf.h_converged_mapped) {
                scf.converged = true;
                scf.iteration = iter;
                break;
            }
        } else if (scf.converged) {
            break;
        }
    }

#ifdef ATV_HAS_HF
    // Post-calculation: one stream fence + host mirrors (not mid-SCF).
    if (device_no_d2h) {
        cudaStreamSynchronize(cu_stream);
        if (scf.h_converged_mapped)
            scf.converged = (*scf.h_converged_mapped != 0);
        if (scf.d_residual) {
            cuda::prec_t h_res = 0;
            cudaMemcpy(&h_res, scf.d_residual, sizeof(cuda::prec_t),
                       cudaMemcpyDeviceToHost);
            scf.residual = static_cast<double>(h_res);
        }
        if (scf.d_V_nuc) {
            cudaMemcpy(&scf.V_nuc, scf.d_V_nuc, sizeof(double),
                       cudaMemcpyDeviceToHost);
        }
    }
#endif

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
    cudaFree(workspace_.d_shell_atom);
    cudaFree(workspace_.d_shell_L);
    cudaFree(workspace_.d_ao_shell);
    cudaFree(workspace_.d_ao_cart);
    cudaFree(workspace_.d_atom_Z);
    cudaFree(workspace_.d_atom_pos);
    cudaFree(workspace_.d_V_nuc);
    cudaFree(workspace_.d_residual);
    if (workspace_.h_converged_mapped) {
        cudaFreeHost(workspace_.h_converged_mapped);
        workspace_.h_converged_mapped = nullptr;
        workspace_.d_converged = nullptr; // alias of mapped host
    }
    cudaFree(workspace_.d_scalar_scratch);
    cudaFree(workspace_.d_W);
    cudaFree(workspace_.C_prime);
    workspace_ = {};

    built_ = false;
}

} // namespace scf
} // namespace cuda
} // namespace ATV

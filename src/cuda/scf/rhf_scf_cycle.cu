/// @file cuda/scf/rhf_scf_cycle.cu
/// @brief internal::RhfScfCycle — one-stage RHF helpers for cuda::scf::RHF.
///
/// The iteration loop lives on `cuda::Driver<scf::Phase>`. Each `step_*`
/// is one phase. Mix is PySCF `scf.diis.CDIIS.update` (device CDIIS).

#include "atomiverse/cuda/scf/internal/rhf_scf_cycle.cuh"

#include <cmath>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

// ── Kernel: de-shift the virtual orbital energies ────────────────────
__global__ void deshift_virtuals_kernel(prec_t* __restrict__ eig, prec_t lambda,
                                        int n_occ, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + n_occ;
    if (i < n)
        eig[i] -= lambda;
}

// Free workspace device arrays (shared destroy epilogue).
static void free_workspace_device(SCFWorkspace& workspace_) {
    cudaFree(workspace_.H0);
    cudaFree(workspace_.H);
    cudaFree(workspace_.P);
    cudaFree(workspace_.P_old);
    cudaFree(workspace_.C);
    cudaFree(workspace_.eig);
    cudaFree(workspace_.work);
    cudaFree(workspace_.Hcore);
    cudaFree(workspace_.S);
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
        workspace_.d_converged = nullptr;
    }
    cudaFree(workspace_.d_scalar_scratch);
    cudaFree(workspace_.d_W);
    cudaFree(workspace_.d_gradient);
    cudaFree(workspace_.C_prime);
    workspace_ = {};
}

RhfScfCycle::RhfScfCycle(xt::xtensor<double, 2> S, xt::xtensor<double, 2> Hcore,
                         xt::xtensor<double, 4> ERI, double V_nuc,
                         int n_electrons, double tol, int max_iter,
                         double level_shift)
    : path_(Path::RHFHostIntegrals), tol_(tol), level_shift_(level_shift) {
    (void)max_iter;
    if (n_electrons < 0 || n_electrons % 2 != 0) {
        throw std::invalid_argument(
            "cuda::scf::RHF: closed-shell RHF requires an even, "
            "non-negative electron count (n_electrons="
            + std::to_string(n_electrons)
            + "). Check the molecular charge, or use the CPU unrestricted "
              "path for open-shell systems.");
    }
    const int n_occ = n_electrons / 2;
    eigen_ = std::make_unique<EigenSolver>();
    convergence_ = std::make_unique<ConvergenceCheck>(tol, false);
    rhf_fock_ =
        std::make_unique<RHFFock>(std::move(S), std::move(Hcore),
                                  std::move(ERI), V_nuc, n_occ, level_shift_);
    rhf_density_ = std::make_unique<RHFDensity>(n_occ);
    diis_ = std::make_unique<DIISMixer>(/*max_vectors=*/8, /*start_iter=*/1,
                                        level_shift_);
}

RhfScfCycle::RhfScfCycle(int n_prim, const double* exponents,
                         const double* coefficients, int n_occ, double tol,
                         int max_iter, double level_shift)
    : RhfScfCycle(/*n_shells_per_atom=*/1, n_prim, exponents, coefficients,
                  n_occ, tol, max_iter, level_shift) {}

RhfScfCycle::RhfScfCycle(int n_shells_per_atom, int n_prim,
                         const double* alpha_atom, const double* coeff_atom,
                         int n_occ, double tol, int max_iter,
                         double level_shift)
    : path_(Path::RHFDeviceIntegrals), tol_(tol), level_shift_(level_shift) {
    (void)max_iter;
    if (n_occ < 0) {
        throw std::invalid_argument(
            "cuda::scf::RHF: n_occ must be non-negative (got "
            + std::to_string(n_occ) + ").");
    }
    if (n_shells_per_atom < 1 || n_prim < 1) {
        throw std::invalid_argument(
            "cuda::scf::RHF: n_shells_per_atom and n_prim must be ≥ 1");
    }
    eigen_ = std::make_unique<EigenSolver>();
    convergence_ = std::make_unique<ConvergenceCheck>(tol, false);
    rhf_fock_device_ = std::make_unique<RHFFockDevice>(
        n_shells_per_atom, n_prim, alpha_atom, coeff_atom, n_occ, level_shift_);
    rhf_density_ = std::make_unique<RHFDensity>(n_occ);
    diis_ = std::make_unique<DIISMixer>(/*max_vectors=*/8, /*start_iter=*/1,
                                        level_shift_);
}

RhfScfCycle::RhfScfCycle(int n_shells, int n_ao, int n_prim,
                         const double* alpha_mol, const double* coeff_mol,
                         const int* L_mol, const int* atom_of_mol, int n_occ,
                         double tol, int max_iter, double level_shift)
    : path_(Path::RHFDeviceIntegrals), tol_(tol), level_shift_(level_shift) {
    (void)max_iter;
    if (n_occ < 0) {
        throw std::invalid_argument(
            "cuda::scf::RHF: n_occ must be non-negative (got "
            + std::to_string(n_occ) + ").");
    }
    if (n_shells < 1 || n_ao < 1 || n_prim < 1) {
        throw std::invalid_argument(
            "cuda::scf::RHF: n_shells, n_ao, n_prim must be ≥ 1");
    }
    eigen_ = std::make_unique<EigenSolver>();
    convergence_ = std::make_unique<ConvergenceCheck>(tol, false);
    rhf_fock_device_ = std::make_unique<RHFFockDevice>(
        n_shells, n_ao, n_prim, alpha_mol, coeff_mol, L_mol, atom_of_mol, n_occ,
        level_shift_);
    rhf_density_ = std::make_unique<RHFDensity>(n_occ);
    diis_ = std::make_unique<DIISMixer>(/*max_vectors=*/8, /*start_iter=*/1,
                                        level_shift_);
}

RhfScfCycle::RhfScfCycle(int n_shells, int n_ao, const double* alpha,
                         const double* coeff, int n_prim_total,
                         const int* nprim, const int* prim_off,
                         const int* L_mol, const int* atom_of_mol, int n_occ,
                         double tol, int max_iter, double level_shift,
                         bool spherical)
    : path_(Path::RHFDeviceIntegrals), tol_(tol), level_shift_(level_shift) {
    (void)max_iter;
    if (n_occ < 0) {
        throw std::invalid_argument(
            "cuda::scf::RHF: n_occ must be non-negative (got "
            + std::to_string(n_occ) + ").");
    }
    if (n_shells < 1 || n_ao < 1 || n_prim_total < 1) {
        throw std::invalid_argument(
            "cuda::scf::RHF: ragged Cartesian shells need n_shells, n_ao, "
            "and primitives ≥ 1");
    }
    eigen_ = std::make_unique<EigenSolver>();
    convergence_ = std::make_unique<ConvergenceCheck>(tol, false);
    rhf_fock_device_ = std::make_unique<RHFFockDevice>(
        n_shells, n_ao, alpha, coeff, n_prim_total, nprim, prim_off, L_mol,
        atom_of_mol, n_occ, level_shift_, spherical);
    rhf_density_ = std::make_unique<RHFDensity>(n_occ);
    diis_ = std::make_unique<DIISMixer>(/*max_vectors=*/8, /*start_iter=*/1,
                                        level_shift_);
}

void RhfScfCycle::build(void* stream, cuda::Frame& frame) {
    SolverContext ctx{stream, frame, 0};
    switch (path_) {
    case Path::RHFHostIntegrals:
        rhf_fock_->build(ctx, workspace_);
        diis_->build(ctx, workspace_);
        eigen_->build(ctx, workspace_);
        rhf_density_->build(ctx, workspace_);
        convergence_->build(ctx, workspace_);
        break;
    case Path::RHFDeviceIntegrals:
        rhf_fock_device_->build(ctx, workspace_);
        diis_->build(ctx, workspace_);
        eigen_->build(ctx, workspace_);
        rhf_density_->build(ctx, workspace_);
        convergence_->build(ctx, workspace_);
        break;
    }
    built_ = true;
}

void RhfScfCycle::begin(void* stream) {
    if (!built_) {
        throw std::runtime_error(
            "cuda::scf::RHF: RhfScfCycle::begin before build()");
    }
    auto cu_stream = static_cast<cudaStream_t>(stream);
    auto& scf = workspace_;
    scf.converged = false;
    scf.iteration = 0;
    polishing_ = false;
    canonicalized_ = false;
    if (scf.h_converged_mapped)
        *scf.h_converged_mapped = 0;
    if (scf.n > 0 && scf.P)
        cudaMemsetAsync(scf.P, 0, scf.n * scf.n * sizeof(cuda::prec_t),
                        cu_stream);
    if (diis_)
        diis_->reset();
    convergence_->reset(stream);
}

void RhfScfCycle::step_fock(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    if (path_ == Path::RHFHostIntegrals)
        rhf_fock_->step(ctx, workspace_);
    else
        rhf_fock_device_->step(ctx, workspace_);
}

void RhfScfCycle::step_mix(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    if (polishing_)
        diis_->canonicalize(ctx, workspace_);
    else
        diis_->step(ctx, workspace_);
}

void RhfScfCycle::step_diagonalize(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    eigen_->step(ctx, workspace_);
}

void RhfScfCycle::step_density(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    rhf_density_->step(ctx, workspace_);
}

void RhfScfCycle::step_check(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    if (path_ == Path::RHFDeviceIntegrals)
        convergence_->step_hf(ctx, workspace_, diis_->error());
    else
        convergence_->step(ctx, workspace_);
}

bool RhfScfCycle::poll_converged(void* stream, int step) {
    auto& scf = workspace_;
    if (path_ == Path::RHFDeviceIntegrals) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        const bool accepted =
            scf.h_converged_mapped && *scf.h_converged_mapped != 0;
        if (accepted && !polishing_) {
            // A candidate must survive one ordinary scheduled iteration with
            // the fresh physical Fock and no DIIS/level shift. Poll changes
            // control only; the existing five stages own all science.
            polishing_ = true;
            *scf.h_converged_mapped = 0;
            scf.converged = false;
            return false;
        }
        canonicalized_ = polishing_ && accepted;
        polishing_ = false;
        scf.converged = accepted;
        if (scf.converged)
            scf.iteration = step;
    }
    return scf.converged;
}

void RhfScfCycle::finish(void* stream) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    auto& scf = workspace_;
    if (path_ == Path::RHFDeviceIntegrals) {
        if (scf.h_converged_mapped)
            scf.converged = (*scf.h_converged_mapped != 0);
    }

    if (scf.converged && !canonicalized_ && std::abs(level_shift_) > 1e-4
        && scf.n_occ < scf.n) {
        const int n_virt = scf.n - scf.n_occ;
        int blocks = cuda::grid_size(n_virt, cuda::kDefaultBlock);
        deshift_virtuals_kernel<<<blocks, cuda::kDefaultBlock, 0, cu_stream>>>(
            scf.eig, static_cast<cuda::prec_t>(level_shift_), scf.n_occ, scf.n);
    }
}

void RhfScfCycle::destroy(void* stream, cuda::Frame& frame) {
    SolverContext ctx{stream, frame, 0};
    switch (path_) {
    case Path::RHFHostIntegrals:
        convergence_->destroy(ctx, workspace_);
        rhf_density_->destroy(ctx, workspace_);
        eigen_->destroy(ctx, workspace_);
        diis_->destroy(ctx, workspace_);
        rhf_fock_->destroy(ctx, workspace_);
        break;
    case Path::RHFDeviceIntegrals:
        convergence_->destroy(ctx, workspace_);
        rhf_density_->destroy(ctx, workspace_);
        eigen_->destroy(ctx, workspace_);
        diis_->destroy(ctx, workspace_);
        rhf_fock_device_->destroy(ctx, workspace_);
        break;
    }
    free_workspace_device(workspace_);
    built_ = false;
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

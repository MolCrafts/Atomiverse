/// @file cuda/scf/uhf_scf_cycle.cu
/// @brief One-stage UHF helpers. The loop is cuda::scf::UHF.

#include "atomiverse/cuda/scf/internal/uhf_scf_cycle.cuh"

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "atomiverse/cuda/scf/internal/diis_mixer.cuh"
#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"
#include "atomiverse/cuda/scf/internal/uhf_density.cuh"
#include "atomiverse/cuda/scf/internal/uhf_fock.cuh"
#include "atomiverse/cuda/scf/internal/uhf_guess.cuh"
#include "cublas_dispatch.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {
namespace {

__global__ void uhf_deshift_virtuals_kernel(prec_t* __restrict__ eig,
                                            prec_t lambda, int n_occ, int n) {
    const int i =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + n_occ;
    if (i < n)
        eig[i] -= lambda;
}

void swap_beta(SCFWorkspace& ws, UhfBeta& b) {
    std::swap(ws.H, b.H);
    std::swap(ws.P, b.P);
    std::swap(ws.P_old, b.P_old);
    std::swap(ws.C, b.C);
    std::swap(ws.C_prime, b.C_prime);
    std::swap(ws.eig, b.eig);
    std::swap(ws.work, b.work);
    std::swap(ws.F_prime, b.F_prime);
    std::swap(ws.n_occ, b.n_occ);
}

struct BetaGuard {
    SCFWorkspace& ws;
    UhfBeta& b;
    explicit BetaGuard(SCFWorkspace& ws_in, UhfBeta& b_in)
        : ws(ws_in), b(b_in) {
        swap_beta(ws, b);
    }
    ~BetaGuard() { swap_beta(ws, b); }
};

void free_beta(UhfBeta& b) {
    cudaFree(b.H);
    cudaFree(b.P);
    cudaFree(b.P_old);
    cudaFree(b.C);
    cudaFree(b.C_prime);
    cudaFree(b.eig);
    cudaFree(b.work);
    cudaFree(b.F_prime);
    b = {};
}

void free_workspace(SCFWorkspace& ws) {
    cudaFree(ws.H0);
    cudaFree(ws.H);
    cudaFree(ws.P);
    cudaFree(ws.P_old);
    cudaFree(ws.C);
    cudaFree(ws.eig);
    cudaFree(ws.work);
    cudaFree(ws.Hcore);
    cudaFree(ws.S);
    cudaFree(ws.X);
    cudaFree(ws.F_prime);
    cudaFree(ws.ERI);
    cudaFree(ws.d_shell_alpha);
    cudaFree(ws.d_shell_coeff);
    cudaFree(ws.d_shell_center);
    cudaFree(ws.d_shell_atom);
    cudaFree(ws.d_shell_L);
    cudaFree(ws.d_ao_shell);
    cudaFree(ws.d_ao_cart);
    cudaFree(ws.d_atom_Z);
    cudaFree(ws.d_atom_pos);
    cudaFree(ws.d_V_nuc);
    cudaFree(ws.d_residual);
    if (ws.h_converged_mapped) {
        cudaFreeHost(ws.h_converged_mapped);
        ws.h_converged_mapped = nullptr;
        ws.d_converged = nullptr;
    }
    cudaFree(ws.d_scalar_scratch);
    cudaFree(ws.d_W);
    cudaFree(ws.d_gradient);
    cudaFree(ws.C_prime);
    ws = {};
}

} // namespace

UhfScfCycle::UhfScfCycle(int n_shells, int n_ao, int n_prim,
                         const double* alpha_mol, const double* coeff_mol,
                         const int* L_mol, const int* atom_of_mol, int n_alpha,
                         int n_beta, double tol, int max_iter,
                         double level_shift, const int* atomic_numbers,
                         int natm)
    : tol_(tol), level_shift_(level_shift) {
    (void)max_iter;
    if (n_alpha < 0 || n_beta < 0) {
        throw std::invalid_argument(
            "cuda::scf::UHF: n_alpha and n_beta must be >= 0 (got "
            + std::to_string(n_alpha) + ", " + std::to_string(n_beta) + ").");
    }
    if (n_shells < 1 || n_ao < 1 || n_prim < 1) {
        throw std::invalid_argument(
            "cuda::scf::UHF: n_shells, n_ao, n_prim must be >= 1");
    }
    eigen_ = std::make_unique<EigenSolver>();
    diis_alpha_ = std::make_unique<DIISMixer>(8, 1, level_shift_);
    diis_beta_ = std::make_unique<DIISMixer>(8, 1, level_shift_);
    fock_ = std::make_unique<UHFFockDevice>(n_shells, n_ao, n_prim, alpha_mol,
                                            coeff_mol, L_mol, atom_of_mol,
                                            n_alpha, n_beta);
    density_ = std::make_unique<UHFDensity>();
    convergence_ = std::make_unique<ConvergenceCheck>(tol, false);
    guess_ = std::make_unique<UhfMinaoGuess>(atomic_numbers, natm);
}

UhfScfCycle::~UhfScfCycle() = default;

void UhfScfCycle::build(void* stream, cuda::Frame& frame) {
    SolverContext ctx{stream, frame, 0};
    fock_->build(ctx, workspace_, beta_);
    diis_alpha_->build(ctx, workspace_);
    {
        BetaGuard guard(workspace_, beta_);
        diis_beta_->build(ctx, workspace_);
    }
    eigen_->build(ctx, workspace_);
    density_->build(ctx, workspace_);
    convergence_->build(ctx, workspace_);
    guess_->build(workspace_, static_cast<cudaStream_t>(stream));
    built_ = true;
}

void UhfScfCycle::begin(void* stream) {
    if (!built_) {
        throw std::runtime_error(
            "cuda::scf::UHF: UhfScfCycle::begin before build()");
    }
    auto cu = static_cast<cudaStream_t>(stream);
    workspace_.converged = false;
    workspace_.iteration = 0;
    polishing_ = false;
    canonicalized_ = false;
    if (workspace_.h_converged_mapped)
        *workspace_.h_converged_mapped = 0;
    const size_t bytes =
        static_cast<size_t>(workspace_.n * workspace_.n) * sizeof(prec_t);
    if (workspace_.P)
        cudaMemsetAsync(workspace_.P, 0, bytes, cu);
    if (beta_.P)
        cudaMemsetAsync(beta_.P, 0, bytes, cu);
    if (diis_alpha_)
        diis_alpha_->reset();
    if (diis_beta_)
        diis_beta_->reset();
    convergence_->reset(stream);
    guess_->seed(workspace_, beta_.P, cu);
}

void UhfScfCycle::step_fock(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    fock_->step(ctx, workspace_, beta_);
}

void UhfScfCycle::step_mix(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    if (polishing_)
        diis_alpha_->canonicalize(ctx, workspace_);
    else
        diis_alpha_->step(ctx, workspace_);
    BetaGuard guard(workspace_, beta_);
    if (polishing_)
        diis_beta_->canonicalize(ctx, workspace_);
    else
        diis_beta_->step(ctx, workspace_);
}

void UhfScfCycle::step_diagonalize(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    eigen_->step(ctx, workspace_);
    BetaGuard guard(workspace_, beta_);
    eigen_->step(ctx, workspace_);
}

void UhfScfCycle::step_density(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    density_->step(ctx, workspace_, beta_);
}

void UhfScfCycle::step_check(void* stream, cuda::Frame& frame, int step) {
    SolverContext ctx{stream, frame, step};
    convergence_->step_hf(ctx, workspace_, diis_alpha_->error(),
                          diis_beta_->error(), beta_.P, beta_.P_old);
}

bool UhfScfCycle::poll_converged(void* stream, int step) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    const bool accepted =
        workspace_.h_converged_mapped && *workspace_.h_converged_mapped != 0;
    if (accepted && !polishing_) {
        polishing_ = true;
        *workspace_.h_converged_mapped = 0;
        workspace_.converged = false;
        return false;
    }
    canonicalized_ = polishing_ && accepted;
    polishing_ = false;
    workspace_.converged = accepted;
    if (workspace_.converged)
        workspace_.iteration = step;
    return workspace_.converged;
}

void UhfScfCycle::finish(void* stream) {
    auto cu = static_cast<cudaStream_t>(stream);
    auto& ws = workspace_;
    if (ws.h_converged_mapped)
        ws.converged = (*ws.h_converged_mapped != 0);
    if (ws.converged && !canonicalized_ && std::abs(level_shift_) > 1e-4) {
        auto deshift = [&](prec_t* eig, int n_occ) {
            if (!eig || n_occ >= ws.n || n_occ <= 0)
                return;
            const int n_virt = ws.n - n_occ;
            uhf_deshift_virtuals_kernel<<<cuda::grid_size(n_virt,
                                                          cuda::kDefaultBlock),
                                          cuda::kDefaultBlock, 0, cu>>>(
                eig, static_cast<prec_t>(level_shift_), n_occ, ws.n);
        };
        deshift(ws.eig, ws.n_occ);
        deshift(beta_.eig, beta_.n_occ);
    }
}

void UhfScfCycle::destroy(void* stream, cuda::Frame& frame) {
    SolverContext ctx{stream, frame, 0};
    if (density_)
        density_->destroy(ctx, workspace_);
    if (eigen_)
        eigen_->destroy(ctx, workspace_);
    if (diis_beta_) {
        BetaGuard guard(workspace_, beta_);
        diis_beta_->destroy(ctx, workspace_);
    }
    if (diis_alpha_)
        diis_alpha_->destroy(ctx, workspace_);
    if (fock_)
        fock_->destroy(ctx, workspace_);
    if (convergence_)
        convergence_->destroy(ctx, workspace_);
    free_beta(beta_);
    free_workspace(workspace_);
    built_ = false;
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

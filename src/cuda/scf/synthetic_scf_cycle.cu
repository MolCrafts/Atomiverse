/// @file cuda/scf/synthetic_scf_cycle.cu
/// @brief internal::SyntheticScfCycle — toy SCF for cuda::scf::SCF tests.

#include "atomiverse/cuda/scf/internal/synthetic_scf_cycle.cuh"

#include <cuda_runtime.h>

#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {

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
    cudaFree(workspace_.C_prime);
    workspace_ = {};
}

SyntheticScfCycle::SyntheticScfCycle(int n_occ, double alpha, double tol,
                                     int max_iter)
    : tol_(tol), max_iter_(max_iter) {
    eigen_ = std::make_unique<EigenSolver>();
    convergence_ = std::make_unique<ConvergenceCheck>(tol, true);
    hamiltonian_ = std::make_unique<Hamiltonian>(n_occ, alpha);
    density_ = std::make_unique<DensityBuilder>();
}

void SyntheticScfCycle::build(void* stream, cuda::Frame& frame) {
    SolverContext ctx{stream, frame, 0};
    hamiltonian_->build(ctx, workspace_);
    eigen_->build(ctx, workspace_);
    density_->build(ctx, workspace_);
    convergence_->build(ctx, workspace_);
    built_ = true;
}

void SyntheticScfCycle::drive(void* stream, cuda::Frame& frame) {
    if (!built_)
        build(stream, frame);

    auto cu_stream = static_cast<cudaStream_t>(stream);
    auto& scf = workspace_;

    scf.converged = false;
    scf.iteration = 0;
    if (scf.n > 0)
        cudaMemsetAsync(scf.P, 0, scf.n * scf.n * sizeof(cuda::prec_t),
                        cu_stream);

    iterations_ = 0;
    // Synthetic: fock(H) → diagonalize → density → check (no mix stage).
    for (int iter = 0; iter < max_iter_; ++iter) {
        SolverContext ctx{stream, frame, /*step=*/iter};
        hamiltonian_->step(ctx, workspace_);
        eigen_->step(ctx, workspace_);
        density_->step(ctx, workspace_);
        convergence_->step(ctx, workspace_);
        ++iterations_;
        if (scf.converged)
            break;
    }
}

void SyntheticScfCycle::destroy(void* stream, cuda::Frame& frame) {
    SolverContext ctx{stream, frame, 0};
    convergence_->destroy(ctx, workspace_);
    density_->destroy(ctx, workspace_);
    eigen_->destroy(ctx, workspace_);
    hamiltonian_->destroy(ctx, workspace_);
    free_workspace_device(workspace_);
    built_ = false;
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

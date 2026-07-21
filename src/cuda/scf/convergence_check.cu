/// @file cuda/scf/convergence_check.cu
/// @brief internal::ConvergenceCheck: ||P - P_old||_F, energy = trace(H*P),
///        convergence flag.

#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "cublas_dispatch.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace scf {
namespace internal {

void ConvergenceCheck::build(const SolverContext& ctx, SCFWorkspace& /*ws*/) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublas_handle_ = handle;
}

void ConvergenceCheck::step(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged)
        return;

    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    int n = scf.n;
    int nn = n * n;

    // work = P - P_old
    cudaMemcpyAsync(scf.work, scf.P, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);
    prec_t neg_one = prec_t(-1);
    detail::axpy(handle, nn, &neg_one, scf.P_old, 1, scf.work, 1);

    // ||P - P_old||_F = nrm2(work)
    prec_t norm = 0;
    detail::nrm2(handle, nn, scf.work, 1, &norm);

    // Optionally compute energy = dot(H, P) (toy-scf mode)
    if (compute_energy_) {
        prec_t energy = 0;
        detail::dot(handle, nn, scf.H, 1, scf.P, 1, &energy);
        cudaMemcpyAsync(&ctx.frame.scalars->energy, &energy, sizeof(prec_t),
                        cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    // Update host-cached convergence state
    scf.residual = static_cast<double>(norm);
    scf.iteration = ctx.step;

    if (ctx.step > 0 && static_cast<double>(norm) < tol_) {
        scf.converged = true;
    }
}

void ConvergenceCheck::destroy(const SolverContext& /*ctx*/,
                               SCFWorkspace& /*ws*/) {
    if (cublas_handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
        cublas_handle_ = nullptr;
    }
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

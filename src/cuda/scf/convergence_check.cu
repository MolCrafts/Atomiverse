/// @file cuda/scf/convergence_check.cu
/// @brief internal::ConvergenceCheck: ||P - P_old||_F on device (no D2H).

#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "cublas_dispatch.cuh"

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace scf {
namespace internal {

/// *d_converged = 1 when step>0 and residual < tol (device-only control).
__global__ void update_scf_converged_kernel(int* __restrict__ d_converged,
                                            const prec_t* __restrict__ residual,
                                            prec_t tol, int step) {
    if (step > 0 && (*residual) < tol)
        *d_converged = 1;
}

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
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    prec_t neg_one = prec_t(-1);
    detail::axpy(handle, nn, &neg_one, scf.P_old, 1, scf.work, 1);

    // Device residual path (RHF device integrals): no D2H of residual.
    if (scf.d_residual && scf.d_converged) {
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        detail::nrm2(handle, nn, scf.work, 1, scf.d_residual);
        update_scf_converged_kernel<<<1, 1, 0, stream>>>(
            scf.d_converged, scf.d_residual, static_cast<prec_t>(tol_),
            ctx.step);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        // Host scf.converged is filled by the RHF criterion (poll_converged)
        // after this stage. The device path does not D2H the residual
        // matrix; the mapped flag is what ends the Driver loop.
        if (compute_energy_ && ctx.frame.scalars) {
            // Toy path should not mix with d_residual device control usually;
            // keep host energy write for synthetic if requested.
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            prec_t energy = 0;
            detail::dot(handle, nn, scf.H, 1, scf.P, 1, &energy);
            energy *= static_cast<prec_t>(units::kHartreeInEv);
            const double frame_energy = static_cast<double>(energy);
            ATV_CUDA_CHECK(cudaMemcpyAsync(&ctx.frame.scalars->energy,
                                           &frame_energy, sizeof(frame_energy),
                                           cudaMemcpyHostToDevice, stream));
        }
        return;
    }

    // Host residual path (synthetic / host-integral uploads).
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    prec_t norm = 0;
    detail::nrm2(handle, nn, scf.work, 1, &norm);

    if (compute_energy_) {
        prec_t energy = 0;
        detail::dot(handle, nn, scf.H, 1, scf.P, 1, &energy);
        energy *= static_cast<prec_t>(units::kHartreeInEv);
        const double frame_energy = static_cast<double>(energy);
        ATV_CUDA_CHECK(cudaMemcpyAsync(&ctx.frame.scalars->energy,
                                       &frame_energy, sizeof(frame_energy),
                                       cudaMemcpyHostToDevice, stream));
    }
    // Single host-mode fence for residual visibility (legacy paths only).
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));

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

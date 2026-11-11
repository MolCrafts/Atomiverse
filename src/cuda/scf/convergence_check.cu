/// @file cuda/scf/convergence_check.cu
/// @brief internal::ConvergenceCheck: ||P - P_old||_F on device (no D2H).

#include "atomiverse/cuda/scf/internal/convergence_check.cuh"
#include "cublas_dispatch.cuh"

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace scf {
namespace internal {

namespace {

void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("CUDA SCF convergence cuBLAS status "
                                 + std::to_string(static_cast<int>(status)));
}

// Scalar control/history only: one thread owns this six-scalar state. All
// matrix reductions feeding it are device-pointer cuBLAS nrm2 calls.
__global__ void reset_hf_convergence(prec_t* metrics) {
    metrics[0] = nan("");
    for (int i = 1; i < 6; ++i)
        metrics[i] = 0;
}

__global__ void update_hf_converged(int* flag, prec_t* residual,
                                    prec_t* metrics,
                                    const SystemScalars* scalars, prec_t tol,
                                    int step) {
    const prec_t energy_ev = scalars->energy;
    const prec_t delta_ha = fabs(energy_ev - metrics[0]) / units::kHartreeInEv;
    const prec_t density = hypot(metrics[1], metrics[2]);
    const prec_t orbital = hypot(metrics[3], metrics[4]);
    metrics[0] = energy_ev;
    metrics[5] = delta_ha;
    *residual = orbital;
    *flag = step > 0 && isfinite(energy_ev) && isfinite(delta_ha)
            && isfinite(density) && isfinite(orbital) && density < tol
            && delta_ha < tol && orbital < sqrt(tol);
}

} // namespace

/// *d_converged = 1 when step>0 and residual < tol (device-only control).
__global__ void update_scf_converged_kernel(int* __restrict__ d_converged,
                                            const prec_t* __restrict__ residual,
                                            prec_t tol, int step) {
    if (step > 0 && (*residual) < tol)
        *d_converged = 1;
}

void ConvergenceCheck::build(const SolverContext& ctx, SCFWorkspace& ws) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    if (cublas_handle_)
        throw std::logic_error(
            "CUDA SCF convergence built twice without destroy");
    if (!(tol_ > 0) || !std::isfinite(tol_) || ws.n <= 0
        || ws.n > std::numeric_limits<int>::max() / ws.n)
        throw std::invalid_argument("CUDA SCF convergence requires finite "
                                    "positive tol and valid matrix order");
    n_ = ws.n;
    try {
        cublasHandle_t handle = nullptr;
        check_blas(cublasCreate(&handle));
        cublas_handle_ = handle;
        check_blas(cublasSetStream(handle, stream));
        if (!compute_energy_) {
            ATV_CUDA_CHECK(
                cudaMalloc(&d_difference_,
                           static_cast<std::size_t>(n_) * n_ * sizeof(prec_t)));
            ATV_CUDA_CHECK(cudaMalloc(&d_metrics_, 6 * sizeof(prec_t)));
            reset(ctx.stream);
        }
    } catch (...) {
        destroy(ctx, ws);
        throw;
    }
}

void ConvergenceCheck::reset(void* stream) {
    if (d_metrics_) {
        reset_hf_convergence<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            d_metrics_);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

void ConvergenceCheck::step_hf(const SolverContext& ctx, SCFWorkspace& ws,
                               const prec_t* error_alpha,
                               const prec_t* error_beta,
                               const prec_t* density_beta,
                               const prec_t* old_density_beta) {
    if (ws.converged)
        return;
    const bool beta = error_beta != nullptr;
    if (!cublas_handle_ || !d_difference_ || !d_metrics_ || ws.n != n_ || !ws.P
        || !ws.P_old || !error_alpha || !ws.d_residual || !ws.d_converged
        || !ctx.frame.scalars || beta != (density_beta != nullptr)
        || beta != (old_density_beta != nullptr))
        throw std::invalid_argument(
            "CUDA HF convergence requires prepared density, physical error and "
            "device control storage");
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    check_blas(cublasSetStream(handle, stream));
    const int nn = n_ * n_;
    const std::size_t bytes = static_cast<std::size_t>(nn) * sizeof(prec_t);
    const prec_t negative_one = -1;
    const auto density_norm = [&](const prec_t* density, const prec_t* previous,
                                  prec_t* output) {
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_difference_, density, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        check_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        check_blas(detail::axpy(handle, nn, &negative_one, previous, 1,
                                d_difference_, 1));
        check_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        check_blas(detail::nrm2(handle, nn, d_difference_, 1, output));
    };
    density_norm(ws.P, ws.P_old, d_metrics_ + 1);
    if (beta)
        density_norm(density_beta, old_density_beta, d_metrics_ + 2);
    else
        ATV_CUDA_CHECK(
            cudaMemsetAsync(d_metrics_ + 2, 0, sizeof(prec_t), stream));
    check_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    check_blas(detail::nrm2(handle, nn, error_alpha, 1, d_metrics_ + 3));
    if (beta)
        check_blas(detail::nrm2(handle, nn, error_beta, 1, d_metrics_ + 4));
    else
        ATV_CUDA_CHECK(
            cudaMemsetAsync(d_metrics_ + 4, 0, sizeof(prec_t), stream));
    update_hf_converged<<<1, 1, 0, stream>>>(
        ws.d_converged, ws.d_residual, d_metrics_, ctx.frame.scalars,
        static_cast<prec_t>(tol_), ctx.step);
    ATV_CUDA_CHECK(cudaGetLastError());
    check_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
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
            cudaMemcpyAsync(&ctx.frame.scalars->energy, &energy, sizeof(prec_t),
                            cudaMemcpyHostToDevice, stream);
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
        cudaMemcpyAsync(&ctx.frame.scalars->energy, &energy, sizeof(prec_t),
                        cudaMemcpyHostToDevice, stream);
    }
    // Single host-mode fence for residual visibility (legacy paths only).
    cudaStreamSynchronize(stream);

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
    cudaFree(d_difference_);
    cudaFree(d_metrics_);
    d_difference_ = d_metrics_ = nullptr;
    n_ = 0;
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

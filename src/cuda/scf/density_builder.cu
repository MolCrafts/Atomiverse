/// @file cuda/scf/density_builder.cu
/// @brief internal::DensityBuilder: P_old = P, then P = C_occ * C_occ^T
///        via cuBLAS gemm.

#include "atomiverse/cuda/scf/internal/density_builder.cuh"
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

void DensityBuilder::build(const SolverContext& ctx, SCFWorkspace& /*ws*/) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublas_handle_ = handle;
}

void DensityBuilder::step(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged)
        return;

    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    int n = scf.n;
    int n_occ = scf.n_occ;

    // Save P -> P_old
    cudaMemcpyAsync(scf.P_old, scf.P, n * n * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);

    // P = C_occ * C_occ^T  (gemm: C * C^T, using first n_occ columns)
    // C is n x n column-major. C_occ = C[:, 0:n_occ] = first n_occ*n elements.
    prec_t one = prec_t(1);
    prec_t zero = prec_t(0);
    detail::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n_occ, &one, scf.C, n,
                 scf.C, n, &zero, scf.P, n);
}

void DensityBuilder::destroy(const SolverContext& /*ctx*/,
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

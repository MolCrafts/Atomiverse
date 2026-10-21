/// @file cuda/scf/eigen_solver.cu
/// @brief internal::EigenSolver: dense symmetric eigendecomposition via
///        cuSOLVER syevd.

#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"
#include "cublas_dispatch.cuh"

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ATV {

using cuda::prec_t;

namespace cuda {
namespace scf {
namespace internal {

namespace {
void check_solver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuda::scf::EigenSolver: ")
                                 + operation + " failed with status "
                                 + std::to_string(static_cast<int>(status)));
}
} // namespace

void EigenSolver::build(const SolverContext& ctx, SCFWorkspace& ws) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    int n = ws.n;

    // Create cuSOLVER handle
    cusolverDnHandle_t handle;
    check_solver(cusolverDnCreate(&handle), "cusolverDnCreate");
    cusolver_handle_ = handle;
    check_solver(cusolverDnSetStream(handle, stream), "cusolverDnSetStream");

    // Query workspace size for syevd
    int lwork = 0;
    check_solver(detail::syevd_bufferSize(
                     handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                     n, static_cast<prec_t*>(nullptr), n,
                     static_cast<prec_t*>(nullptr), &lwork),
                 "syevd_bufferSize");

    work_size_ = lwork;
    ATV_CUDA_CHECK(cudaMalloc(&d_work_, std::size_t(lwork) * sizeof(prec_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d_info_, sizeof(int)));
}

void EigenSolver::step(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged)
        return;

    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cusolverDnHandle_t>(cusolver_handle_);
    int n = scf.n;

    // Copy H -> C (syevd solves in-place, overwrites input with eigenvectors)
    ATV_CUDA_CHECK(cudaMemcpyAsync(scf.C, scf.H,
                                   std::size_t(n) * n * sizeof(prec_t),
                                   cudaMemcpyDeviceToDevice, stream));

    // Solve: C -> eigenvectors (column-major), eig -> eigenvalues (ascending)
    check_solver(detail::syevd(handle, CUSOLVER_EIG_MODE_VECTOR,
                               CUBLAS_FILL_MODE_LOWER, n, scf.C, n, scf.eig,
                               d_work_, work_size_, d_info_),
                 "syevd");
}

void EigenSolver::destroy(const SolverContext& /*ctx*/, SCFWorkspace& /*ws*/) {
    if (cusolver_handle_) {
        check_solver(cusolverDnDestroy(
                         static_cast<cusolverDnHandle_t>(cusolver_handle_)),
                     "cusolverDnDestroy");
        cusolver_handle_ = nullptr;
    }
    ATV_CUDA_CHECK(cudaFree(d_work_));
    d_work_ = nullptr;
    ATV_CUDA_CHECK(cudaFree(d_info_));
    d_info_ = nullptr;
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

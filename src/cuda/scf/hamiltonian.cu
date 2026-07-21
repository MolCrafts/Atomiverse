/// @file cuda/scf/hamiltonian.cu
/// @brief internal::Hamiltonian: allocate SCF workspace + build H = H0 + alpha * P.

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/scf/internal/hamiltonian.cuh"
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

// ── Kernel: initialize H0 as a deterministic symmetric matrix ────────

/// Generate a diagonally-dominant symmetric H0.
/// H0[i,j] = 1.0 / (1 + |i-j|)  for off-diagonal
/// H0[i,i] = n + 1 - i           for diagonal (ensures positive-definite)
/// Column-major layout.
__global__ void init_h0_kernel(prec_t* H0, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    for (int k = idx; k < total; k += blockDim.x * gridDim.x) {
        int col = k / n;
        int row = k % n;
        if (row == col) {
            H0[k] = static_cast<prec_t>(n + 1 - row);
        } else {
            int diff = row > col ? row - col : col - row;
            H0[k] = static_cast<prec_t>(prec_t(1) / prec_t(1 + diff));
        }
    }
}

// ── Lifecycle ────────────────────────────────────────────────────────

void Hamiltonian::build(const SolverContext& ctx, SCFWorkspace& ws) {
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto& scf = ws;
    int n = ctx.frame.n_atoms();

    // Store metadata
    scf.n = n;
    scf.n_occ = n_occ_;
    scf.converged = false;
    scf.iteration = 0;
    scf.residual = 0.0; // residual is double, not work_t

    int nn = n * n;

    // Allocate all SCF workspace matrices
    cudaMalloc(&scf.H0, nn * sizeof(prec_t));
    cudaMalloc(&scf.H, nn * sizeof(prec_t));
    cudaMalloc(&scf.P, nn * sizeof(prec_t));
    cudaMalloc(&scf.P_old, nn * sizeof(prec_t));
    cudaMalloc(&scf.C, nn * sizeof(prec_t));
    cudaMalloc(&scf.eig, n * sizeof(prec_t));
    cudaMalloc(&scf.work, nn * sizeof(prec_t));

    // Zero-initialize P
    cudaMemsetAsync(scf.P, 0, nn * sizeof(prec_t), stream);
    cudaMemsetAsync(scf.P_old, 0, nn * sizeof(prec_t), stream);

    // Initialize H0
    int threads = cuda::kDefaultBlock;
    int blocks = cuda::grid_size(nn, threads);
    init_h0_kernel<<<blocks, threads, 0, stream>>>(scf.H0, n);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublas_handle_ = handle;

    cudaStreamSynchronize(stream);
}

void Hamiltonian::step(const SolverContext& ctx, SCFWorkspace& ws) {
    auto& scf = ws;
    if (scf.converged)
        return;

    auto stream = static_cast<cudaStream_t>(ctx.stream);
    auto handle = static_cast<cublasHandle_t>(cublas_handle_);
    int n = scf.n;
    int nn = n * n;

    // H = H0 (copy)
    cudaMemcpyAsync(scf.H, scf.H0, nn * sizeof(prec_t),
                    cudaMemcpyDeviceToDevice, stream);

    // H += alpha * P
    prec_t alpha = static_cast<prec_t>(alpha_);
    detail::axpy(handle, nn, &alpha, scf.P, 1, scf.H, 1);
}

void Hamiltonian::destroy(const SolverContext& /*ctx*/, SCFWorkspace& /*ws*/) {
    if (cublas_handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
        cublas_handle_ = nullptr;
    }
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

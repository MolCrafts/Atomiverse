/// @file cuda/linalg/pulay.cu
/// @brief Device bordered Pulay LU, moved out of the SCF DIIS mixer.

#include "atomiverse/cuda/linalg/pulay.cuh"

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"

#include <stdexcept>
#include <string>

namespace ATV::cuda::linalg {
namespace {

/// @brief Assemble [Gram,-1;-1,0] and rhs (0,...,0,-1), column major.
///
/// Elementwise map: one thread per entry, contiguous column-major writes.
/// No reduction or shared storage; cuBLAS computed the error inner products.
/// Physical history slots are sufficient: a permutation of Pulay rows and
/// columns leaves the extrapolation invariant, including circular overwrite.
__global__ void assemble_pulay_kernel(const prec_t* gram, prec_t* A,
                                      prec_t* rhs, int m) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dim = m + 1;
    if (idx >= dim * dim)
        return;
    const int row = idx % dim;
    const int col = idx / dim;
    A[idx] = row < m && col < m     ? gram[row + col * m]
             : row == m && col == m ? prec_t(0)
                                    : prec_t(-1);
    if (col == 0)
        rhs[row] = row == m ? prec_t(-1) : prec_t(0);
}

/// @brief On a singular/invalid solve select the most recent vector exactly.
///
/// This tiny control operation inspects at most max_vectors+1 LU pivots and
/// coefficients. A single thread avoids a second scratch reduction/launch;
/// dense matrix science is performed by cuSOLVER, not here.
__global__ void validate_pulay_kernel(const prec_t* lu, prec_t* c,
                                      const int* info, int m, int latest) {
    const int dim = m + 1;
    bool valid = info[0] == 0 && info[1] == 0;
    for (int i = 0; i < dim; ++i)
        valid = valid && isfinite(c[i]) && isfinite(lu[i + i * dim])
                && fabs(lu[i + i * dim]) >= 1e-14;
    if (!valid)
        for (int i = 0; i < m; ++i)
            c[i] = i == latest ? prec_t(1) : prec_t(0);
}

void check_solver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuda::linalg Pulay cuSOLVER status "
                                 + std::to_string(static_cast<int>(status)));
}

} // namespace

void solve_pulay(cusolverDnHandle_t solver, cudaStream_t stream,
                 const prec_t* gram, prec_t* augmented, prec_t* coefficients,
                 prec_t* workspace, int* pivots, int* info, int m, int latest) {
    const int dim = m + 1;
    assemble_pulay_kernel<<<grid_size(dim * dim, kDefaultBlock), kDefaultBlock,
                            0, stream>>>(gram, augmented, coefficients, m);
    ATV_CUDA_CHECK(cudaGetLastError());
    check_solver(cusolverDnSetStream(solver, stream));
    check_solver(cusolverDnDgetrf(solver, dim, dim, augmented, dim, workspace,
                                  pivots, info));
    check_solver(cusolverDnDgetrs(solver, CUBLAS_OP_N, dim, 1, augmented, dim,
                                  pivots, coefficients, dim, info + 1));
    validate_pulay_kernel<<<1, 1, 0, stream>>>(augmented, coefficients, info, m,
                                               latest);
    ATV_CUDA_CHECK(cudaGetLastError());
}

} // namespace ATV::cuda::linalg

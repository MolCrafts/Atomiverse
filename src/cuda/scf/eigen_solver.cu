/// @file cuda/scf/eigen_solver.cu
/// @brief Device symmetric eigendecomposition with bounded CUDA-only dispatch.
///
/// The small-matrix cuSOLVER Jacobi primitive is reused where measured
/// residency permits it. Larger supported matrices use a device-controlled
/// cyclic Jacobi sweep: the installed library downloads a norm per sweep.

#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"

#include "atomiverse/cuda/device/error.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cusolverDn.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {
namespace scf {
namespace internal {
namespace {
constexpr int kThreads = 256;
constexpr int kLibraryLimit = 64;
constexpr int kDenseLimit = 215;
constexpr int kMaxPairs = (kDenseLimit + 1) / 2;
constexpr int kMaxSweeps = 100;
// Stored off-diagonal entries can be annihilated below this relative norm;
// this is an algorithm stop criterion, not a replacement for physical gates.
constexpr double kRelativeTolerance =
    8 * std::numeric_limits<double>::epsilon();

void check_solver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string("CUDA SCF eigensolver ")
                                 + operation + " failed with status "
                                 + std::to_string(static_cast<int>(status)));
}

__global__ void pack_lower_kernel(const double* H, double* C, int n,
                                  int* status) {
    if (threadIdx.x == 0)
        status[0] = status[1] = 0;
    __syncthreads();
    for (int ij = threadIdx.x; ij < n * n; ij += kThreads) {
        const int i = ij % n, j = ij / n;
        const double value = H[i >= j ? ij : j + i * n];
        C[ij] = value;
        if (!isfinite(value))
            atomicExch(status + 1, n + 2);
    }
}

/// @brief One CTA owns a full cyclic Jacobi solve; all convergence is device-side.
///
/// A round contains disjoint coordinate pairs. Each unordered pair of pairs
/// has one owner that loads its old 2x2 block, applies both rotations and
/// writes it and its transpose. Those blocks never overlap, so no global
/// matrix ping-pong is needed. Eigenvector tasks own one row and one column
/// pair. Barriers separate every round. CUB owns reductions and final sorting.
__global__ void cyclic_jacobi_kernel(const double* H, double* A, double* C,
                                     double* eig, int n, int* status) {
    using Reduce = cub::BlockReduce<double, kThreads>;
    using Sort = cub::BlockRadixSort<double, kThreads, 1, int>;
    __shared__ union {
        typename Reduce::TempStorage reduce;
        typename Sort::TempStorage sort;
    } storage;
    __shared__ int left[kMaxPairs], right[kMaxPairs];
    __shared__ double cosine[kMaxPairs], sine[kMaxPairs], tangent[kMaxPairs];
    __shared__ int permutation[kThreads];
    __shared__ double scale, threshold2;
    __shared__ int done;
    const int tid = threadIdx.x;
    const int nn = n * n;
    if (tid == 0)
        status[0] = status[1] = 0;
    __syncthreads();

    double local_max = 0;
    for (int ij = tid; ij < nn; ij += kThreads) {
        const int i = ij % n, j = ij / n;
        const double value = H[i >= j ? ij : j + i * n];
        if (!isfinite(value))
            atomicExch(status, n + 2);
        else
            local_max = fmax(local_max, fabs(value));
    }
    const double maximum =
        Reduce(storage.reduce).Reduce(local_max, ::cuda::maximum<>{});
    if (tid == 0)
        scale = maximum == 0 ? 1 : maximum;
    __syncthreads();
    if (*status != 0)
        return;

    double local_norm2 = 0;
    for (int ij = tid; ij < nn; ij += kThreads) {
        const int i = ij % n, j = ij / n;
        const double value = H[i >= j ? ij : j + i * n] / scale;
        A[ij] = value;
        C[ij] = i == j ? 1 : 0;
        local_norm2 = fma(value, value, local_norm2);
    }
    const double norm2 = Reduce(storage.reduce).Sum(local_norm2);
    if (tid == 0)
        threshold2 = kRelativeTolerance * kRelativeTolerance * norm2;
    __syncthreads();

    const int padded = n + (n & 1);
    const int pairs = padded / 2;
    for (int sweep = 0; sweep <= kMaxSweeps; ++sweep) {
        double off2 = 0;
        for (int ij = tid; ij < nn; ij += kThreads) {
            const int i = ij % n, j = ij / n;
            const double value = A[ij];
            if (!isfinite(value))
                atomicExch(status, n + 2);
            else if (i != j)
                off2 = fma(value, value, off2);
        }
        const double off_norm2 = Reduce(storage.reduce).Sum(off2);
        if (tid == 0)
            done = off_norm2 <= threshold2;
        __syncthreads();
        if (*status != 0)
            return;
        if (done || sweep == kMaxSweeps)
            break;

        for (int round = 0; round < padded - 1; ++round) {
            if (tid < pairs) {
                const int i =
                    tid == 0 ? padded - 1 : (round + tid) % (padded - 1);
                const int j = tid == 0
                                  ? round
                                  : (round - tid + padded - 1) % (padded - 1);
                left[tid] = i;
                right[tid] = j;
                double t = 0, c = 1, s = 0;
                if (i < n && j < n) {
                    const double b = A[i + j * n];
                    if (b != 0) {
                        const double delta =
                            0.5 * (A[j + j * n] - A[i + i * n]);
                        t = b / (delta + copysign(hypot(delta, b), delta));
                        c = 1.0 / sqrt(::fma(t, t, 1.0));
                        s = t * c;
                    }
                }
                tangent[tid] = t;
                cosine[tid] = c;
                sine[tid] = s;
            }
            __syncthreads();

            for (int task = tid; task < pairs * pairs; task += kThreads) {
                const int r = task / pairs, q = task % pairs;
                if (r > q)
                    continue;
                const int i = left[r], j = right[r];
                const int k = left[q], l = right[q];
                if (r == q) {
                    if (i < n && j < n) {
                        const double b = A[i + j * n];
                        A[i + i * n] = fma(-tangent[r], b, A[i + i * n]);
                        A[j + j * n] = fma(tangent[r], b, A[j + j * n]);
                        A[i + j * n] = 0;
                        A[j + i * n] = 0;
                    }
                    continue;
                }
                const double a00 = i < n && k < n ? A[i + k * n] : 0;
                const double a01 = i < n && l < n ? A[i + l * n] : 0;
                const double a10 = j < n && k < n ? A[j + k * n] : 0;
                const double a11 = j < n && l < n ? A[j + l * n] : 0;
                const double cr = cosine[r], sr = sine[r];
                const double cq = cosine[q], sq = sine[q];
                const double t00 = fma(-sr, a10, cr * a00);
                const double t01 = fma(-sr, a11, cr * a01);
                const double t10 = fma(sr, a00, cr * a10);
                const double t11 = fma(sr, a01, cr * a11);
                const double b00 = fma(-sq, t01, cq * t00);
                const double b01 = fma(sq, t00, cq * t01);
                const double b10 = fma(-sq, t11, cq * t10);
                const double b11 = fma(sq, t10, cq * t11);
                if (i < n && k < n)
                    A[i + k * n] = A[k + i * n] = b00;
                if (i < n && l < n)
                    A[i + l * n] = A[l + i * n] = b01;
                if (j < n && k < n)
                    A[j + k * n] = A[k + j * n] = b10;
                if (j < n && l < n)
                    A[j + l * n] = A[l + j * n] = b11;
            }
            // Rows are the fast dimension: adjacent lanes read/write adjacent
            // elements of the two eigenvector columns owned by this pair.
            for (int task = tid; task < n * pairs; task += kThreads) {
                const int row = task % n, pair = task / n;
                const int i = left[pair], j = right[pair];
                const double x = i < n ? C[row + i * n] : 0;
                const double y = j < n ? C[row + j * n] : 0;
                if (i < n)
                    C[row + i * n] = fma(-sine[pair], y, cosine[pair] * x);
                if (j < n)
                    C[row + j * n] = fma(sine[pair], x, cosine[pair] * y);
            }
            __syncthreads();
        }
    }
    if (!done) {
        if (tid == 0)
            *status = n + 1;
        return;
    }

    double key[1] = {tid < n ? A[tid + tid * n] * scale : HUGE_VAL};
    int column[1] = {tid};
    if (tid < n && !isfinite(key[0]))
        atomicExch(status, n + 2);
    __syncthreads();
    if (*status != 0)
        return;
    Sort(storage.sort).Sort(key, column);
    permutation[tid] = column[0];
    if (tid < n)
        eig[tid] = key[0];
    __syncthreads();
    // The Jacobi work matrix is dead after sorting. Reuse it for the
    // eigenvector column permutation instead of a second n*n allocation.
    for (int ij = tid; ij < nn; ij += kThreads) {
        const int row = ij % n, col = ij / n;
        A[ij] = C[row + permutation[col] * n];
    }
    __syncthreads();
    for (int ij = tid; ij < nn; ij += kThreads)
        C[ij] = A[ij];
}
} // namespace

void EigenSolver::build(const SolverContext& ctx, SCFWorkspace& ws) {
    if (d_info_)
        throw std::logic_error("CUDA SCF eigensolver: build called twice");
    if (ws.n <= 0 || ws.n > std::numeric_limits<int>::max() / ws.n)
        throw std::invalid_argument(
            "CUDA SCF eigensolver: invalid matrix order");
    if (!ws.H || !ws.C || !ws.eig)
        throw std::invalid_argument(
            "CUDA SCF eigensolver: missing H/C/eig workspace");
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    n_ = ws.n;
    try {
        if (n_ <= kLibraryLimit || n_ > kDenseLimit) {
            cusolverDnHandle_t handle = nullptr;
            check_solver(cusolverDnCreate(&handle), "cusolverDnCreate");
            cusolver_handle_ = handle;
            check_solver(cusolverDnSetStream(handle, stream),
                         "cusolverDnSetStream");
            if (n_ <= kLibraryLimit) {
                syevjInfo_t params = nullptr;
                check_solver(cusolverDnCreateSyevjInfo(&params),
                             "cusolverDnCreateSyevjInfo");
                jacobi_params_ = params;
                check_solver(cusolverDnXsyevjSetTolerance(params, 1e-14),
                             "cusolverDnXsyevjSetTolerance");
                check_solver(cusolverDnXsyevjSetMaxSweeps(params, kMaxSweeps),
                             "cusolverDnXsyevjSetMaxSweeps");
                check_solver(cusolverDnXsyevjSetSortEig(params, 1),
                             "cusolverDnXsyevjSetSortEig");
                check_solver(cusolverDnDsyevjBatched_bufferSize(
                                 handle, CUSOLVER_EIG_MODE_VECTOR,
                                 CUBLAS_FILL_MODE_LOWER, n_, ws.C, n_, ws.eig,
                                 &library_work_elements_, params, 1),
                             "cusolverDnDsyevjBatched_bufferSize");
            } else {
                // Preserve the generic synthetic helper's original dimension
                // contract. Molecular HF rejects n^4 overflow upstream and
                // cannot enter this legacy library norm-readback path.
                check_solver(cusolverDnDsyevd_bufferSize(
                                 handle, CUSOLVER_EIG_MODE_VECTOR,
                                 CUBLAS_FILL_MODE_LOWER, n_, ws.C, n_, ws.eig,
                                 &library_work_elements_),
                             "cusolverDnDsyevd_bufferSize");
            }
            device_work_bytes_ =
                static_cast<std::size_t>(library_work_elements_)
                * sizeof(double);
        } else {
            device_work_bytes_ =
                static_cast<std::size_t>(n_) * n_ * sizeof(double);
        }
        if (device_work_bytes_ > 0)
            ATV_CUDA_CHECK(cudaMalloc(&d_work_, device_work_bytes_));
        ATV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_info_),
                                     2 * sizeof(int), cudaHostAllocMapped));
        h_info_[0] = h_info_[1] = 0;
        ATV_CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&d_info_), h_info_, 0));
    } catch (...) {
        destroy(ctx, ws);
        throw;
    }
}

void EigenSolver::step(const SolverContext& ctx, SCFWorkspace& ws) {
    if (ws.converged)
        return;
    if (!d_info_ || ws.n != n_ || !ws.H || !ws.C || !ws.eig)
        throw std::logic_error(
            "CUDA SCF eigensolver: missing or changed workspace");
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    if (n_ <= kLibraryLimit || n_ > kDenseLimit) {
        auto handle = static_cast<cusolverDnHandle_t>(cusolver_handle_);
        check_solver(cusolverDnSetStream(handle, stream),
                     "cusolverDnSetStream");
        pack_lower_kernel<<<1, kThreads, 0, stream>>>(ws.H, ws.C, n_, d_info_);
        ATV_CUDA_CHECK(cudaGetLastError());
        if (n_ <= kLibraryLimit) {
            auto params = static_cast<syevjInfo_t>(jacobi_params_);
            check_solver(cusolverDnDsyevjBatched(
                             handle, CUSOLVER_EIG_MODE_VECTOR,
                             CUBLAS_FILL_MODE_LOWER, n_, ws.C, n_, ws.eig,
                             static_cast<double*>(d_work_),
                             library_work_elements_, d_info_, params, 1),
                         "cusolverDnDsyevjBatched");
        } else {
            check_solver(cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_VECTOR,
                                          CUBLAS_FILL_MODE_LOWER, n_, ws.C, n_,
                                          ws.eig, static_cast<double*>(d_work_),
                                          library_work_elements_, d_info_),
                         "cusolverDnDsyevd");
        }
    } else {
        cyclic_jacobi_kernel<<<1, kThreads, 0, stream>>>(
            ws.H, static_cast<double*>(d_work_), ws.C, ws.eig, n_, d_info_);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_info_[0] != 0 || h_info_[1] != 0)
        throw std::runtime_error(
            "CUDA SCF eigensolver failed, info=" + std::to_string(h_info_[0])
            + ", input_status=" + std::to_string(h_info_[1]));
}

void EigenSolver::destroy(const SolverContext&, SCFWorkspace&) {
    if (jacobi_params_)
        cusolverDnDestroySyevjInfo(static_cast<syevjInfo_t>(jacobi_params_));
    jacobi_params_ = nullptr;
    if (cusolver_handle_)
        cusolverDnDestroy(static_cast<cusolverDnHandle_t>(cusolver_handle_));
    cusolver_handle_ = nullptr;
    cudaFree(d_work_);
    d_work_ = nullptr;
    if (h_info_)
        cudaFreeHost(h_info_);
    h_info_ = nullptr;
    d_info_ = nullptr;
    device_work_bytes_ = 0;
    library_work_elements_ = 0;
    n_ = 0;
}

} // namespace internal
} // namespace scf
} // namespace cuda
} // namespace ATV

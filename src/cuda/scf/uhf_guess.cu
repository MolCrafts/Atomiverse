/// @file cuda/scf/uhf_guess.cu
/// @brief Device MINAO projection using cuSOLVER Cholesky and cuBLAS.
#ifdef ATV_HAS_HF

#include "atomiverse/cuda/scf/internal/uhf_guess.cuh"

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/scf/device_integrals.cuh"

#include <limits>
#include <stdexcept>
#include <string>

namespace ATV::cuda::scf::internal {
namespace {

void check_solver(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("CUDA UHF MINAO: cuSOLVER status "
                                 + std::to_string(static_cast<int>(status)));
}

void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("CUDA UHF MINAO: cuBLAS status "
                                 + std::to_string(static_cast<int>(status)));
}

__global__ void weight_minao_columns(double* scaled, const double* projected,
                                     const double* occupation, int n, int m) {
    const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (k < n * m)
        scaled[k] = projected[k] * occupation[k / n];
}

__global__ void publish_minao_spin_density(const double* total, double* alpha,
                                           double* beta, int n, int* bad) {
    const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= n * n)
        return;
    const int i = k % n, j = k / n;
    // CPU symmetrizes P then halves it for each spin. Read an immutable
    // temporary so the transpose reads cannot race with density publication.
    const double value = 0.25 * (total[k] + total[j + i * n]);
    if (!isfinite(value))
        atomicExch(bad, 1);
    alpha[k] = value;
    beta[k] = value;
}

} // namespace

UhfMinaoGuess::UhfMinaoGuess(const int* atomic_numbers, int natm)
    : table_(atomic_numbers, natm), natm_(natm) {}

UhfMinaoGuess::~UhfMinaoGuess() { clear_(); }

void UhfMinaoGuess::clear_() noexcept {
    if (solver_)
        cusolverDnDestroy(solver_);
    if (blas_)
        cublasDestroy(blas_);
    solver_ = nullptr;
    blas_ = nullptr;
    cudaFree(alpha_);
    cudaFree(coefficient_);
    cudaFree(occupation_);
    cudaFree(offset_);
    cudaFree(angular_);
    cudaFree(atom_);
    cudaFree(ao_shell_);
    cudaFree(ao_cart_);
    cudaFree(factor_);
    cudaFree(projected_);
    cudaFree(scaled_);
    cudaFree(potrf_work_);
    if (h_status_)
        cudaFreeHost(h_status_);
    alpha_ = coefficient_ = occupation_ = nullptr;
    offset_ = angular_ = atom_ = ao_shell_ = ao_cart_ = nullptr;
    factor_ = projected_ = scaled_ = potrf_work_ = nullptr;
    h_status_ = d_status_ = nullptr;
    n_ = m_ = potrf_size_ = 0;
}

void UhfMinaoGuess::build(const SCFWorkspace& ws, cudaStream_t stream) {
    if (n_)
        throw std::logic_error("CUDA UHF MINAO: build called twice");
    if (ws.n <= 0 || ws.n > std::numeric_limits<int>::max() / ws.n
        || ws.natm != natm_)
        throw std::invalid_argument("CUDA UHF MINAO: invalid AO dimension");
    if (table_.occupation.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max() / ws.n))
        throw std::overflow_error(
            "CUDA UHF MINAO: projection exceeds index range");
    n_ = ws.n;
    m_ = static_cast<int>(table_.occupation.size());
    if (!m_)
        return;
    try {
        auto upload = [&](auto*& dst, const auto& source) {
            const std::size_t bytes = source.size() * sizeof(source[0]);
            ATV_CUDA_CHECK(cudaMalloc(&dst, bytes));
            ATV_CUDA_CHECK(cudaMemcpyAsync(dst, source.data(), bytes,
                                           cudaMemcpyHostToDevice, stream));
        };
        upload(alpha_, table_.alpha);
        upload(coefficient_, table_.coefficient);
        upload(occupation_, table_.occupation);
        upload(offset_, table_.primitive_offset);
        upload(angular_, table_.angular);
        upload(atom_, table_.atom);
        upload(ao_shell_, table_.ao_shell);
        upload(ao_cart_, table_.ao_cart);
        const std::size_t matrix_bytes =
            static_cast<std::size_t>(n_) * n_ * sizeof(double);
        const std::size_t projection_bytes =
            static_cast<std::size_t>(n_) * m_ * sizeof(double);
        ATV_CUDA_CHECK(cudaMalloc(&factor_, matrix_bytes));
        ATV_CUDA_CHECK(cudaMalloc(&projected_, projection_bytes));
        ATV_CUDA_CHECK(cudaMalloc(&scaled_, projection_bytes));
        check_blas(cublasCreate(&blas_));
        check_blas(cublasSetStream(blas_, stream));
        check_blas(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST));
        check_solver(cusolverDnCreate(&solver_));
        check_solver(cusolverDnSetStream(solver_, stream));
        check_solver(cusolverDnDpotrf_bufferSize(
            solver_, CUBLAS_FILL_MODE_LOWER, n_, factor_, n_, &potrf_size_));
        if (potrf_size_ < 0)
            throw std::runtime_error(
                "CUDA UHF MINAO: invalid solver workspace");
        if (potrf_size_)
            ATV_CUDA_CHECK(
                cudaMalloc(&potrf_work_, static_cast<std::size_t>(potrf_size_)
                                             * sizeof(double)));
        // Only normalization / factorization / solve success bits are visible
        // to the host. All scientific results remain in device storage.
        ATV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_status_),
                                     3 * sizeof(int), cudaHostAllocMapped));
        h_status_[0] = h_status_[1] = h_status_[2] = 0;
        ATV_CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&d_status_), h_status_, 0));
        launch_normalize_minao_sp(alpha_, coefficient_, offset_, angular_,
                                  static_cast<int>(table_.angular.size()),
                                  d_status_, stream);
        ATV_CUDA_CHECK(cudaGetLastError());
    } catch (...) {
        clear_();
        throw;
    }
}

void UhfMinaoGuess::seed(const SCFWorkspace& ws, prec_t* beta_density,
                         cudaStream_t stream) {
    if (!n_ || ws.n != n_ || ws.natm != natm_)
        throw std::logic_error("CUDA UHF MINAO: missing or changed setup");
    if (!m_)
        return;
    if (ws.n_prim <= 0 || !ws.S || !ws.P || !beta_density || !ws.d_shell_alpha
        || !ws.d_shell_coeff || !ws.d_shell_center || !ws.d_shell_L
        || !ws.d_ao_shell || !ws.d_ao_cart || !ws.d_atom_pos)
        throw std::logic_error("CUDA UHF MINAO: incomplete device workspace");
    check_solver(cusolverDnSetStream(solver_, stream));
    check_blas(cublasSetStream(blas_, stream));
    ATV_CUDA_CHECK(cudaMemsetAsync(d_status_ + 1, 0, 2 * sizeof(int), stream));
    launch_overlap_minao_sp(projected_, ws.d_shell_alpha, ws.d_shell_coeff,
                            ws.d_shell_center, ws.d_shell_L, ws.d_ao_shell,
                            ws.d_ao_cart, n_, ws.n_prim, alpha_, coefficient_,
                            offset_, angular_, atom_, ao_shell_, ao_cart_, m_,
                            ws.d_atom_pos, stream);
    ATV_CUDA_CHECK(cudaGetLastError());
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        factor_, ws.S, static_cast<std::size_t>(n_) * n_ * sizeof(double),
        cudaMemcpyDeviceToDevice, stream));
    check_solver(cusolverDnDpotrf(solver_, CUBLAS_FILL_MODE_LOWER, n_, factor_,
                                  n_, potrf_work_, potrf_size_, d_status_ + 1));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_status_[0] || h_status_[1])
        throw std::runtime_error("CUDA UHF MINAO: invalid normalization or "
                                 "non-positive overlap, info="
                                 + std::to_string(h_status_[1]));
    check_solver(cusolverDnDpotrs(solver_, CUBLAS_FILL_MODE_LOWER, n_, m_,
                                  factor_, n_, projected_, n_, d_status_ + 2));
    constexpr int threads = 128;
    weight_minao_columns<<<cuda::grid_size(n_ * m_, threads), threads, 0,
                           stream>>>(scaled_, projected_, occupation_, n_, m_);
    ATV_CUDA_CHECK(cudaGetLastError());
    const double one = 1, zero = 0;
    // The Cholesky factor is dead after potrs, so reuse its matrix storage
    // for P_total rather than materializing another nAO^2 temporary.
    check_blas(cublasDgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_T, n_, n_, m_, &one,
                           scaled_, n_, projected_, n_, &zero, factor_, n_));
    publish_minao_spin_density<<<cuda::grid_size(n_ * n_, threads), threads, 0,
                                 stream>>>(factor_, ws.P, beta_density, n_,
                                           d_status_ + 2);
    ATV_CUDA_CHECK(cudaGetLastError());
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_status_[2])
        throw std::runtime_error(
            "CUDA UHF MINAO: solve or density failed, info="
            + std::to_string(h_status_[2]));
}

} // namespace ATV::cuda::scf::internal
#endif

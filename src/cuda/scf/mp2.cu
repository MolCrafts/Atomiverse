/// @file cuda/scf/mp2.cu
/// @brief Device-only RHF-to-RMP2 handoff, occupied-tiled AO-to-MO and energies.
#ifdef ATV_HAS_HF

#include "atomiverse/cuda/scf/mp2.cuh"

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/scf/internal/mp2_device.cuh"
#include "atomiverse/cuda/scf/rhf.cuh"

namespace ATV::cuda::scf::internal {
namespace {

struct Channels {
    double os;
    double ss;
    unsigned invalid;
};

struct Packet {
    Channels channels;
    double prior_ev;
    double reference;
    double correlation;
    double total;
};

struct AddChannels {
    __host__ __device__ Channels operator()(Channels a, Channels b) const {
        return {a.os + b.os, a.ss + b.ss, a.invalid | b.invalid};
    }
};

// SIMT: CUB owns thread/warp/block/grid reduction. Each mapped lane computes
// one (a,j,b) term from the contiguous a-fast tile; no contribution tensor is
// stored. Energies and products stay FP64. The exchanged integral is a tile
// transpose read; no second ovov array or global atomic is needed.
struct EnergyTerm {
    const double* tile;
    const double* energies;
    int i;
    int n_occ;
    int first_occ;
    int n_active;
    int n_vir;

    __host__ __device__ Channels operator()(int index) const {
        const int a = index % n_vir;
        const int jb = index / n_vir;
        const int j = jb % n_active;
        const int b = jb / n_active;
        const double direct = tile[index];
        const double exchange = tile[b + j * n_vir + a * n_vir * n_active];
        const double denom = energies[i] + energies[first_occ + j]
                             - energies[n_occ + a] - energies[n_occ + b];
        if (!(denom < 0.0) || !isfinite(denom) || !isfinite(direct)
            || !isfinite(exchange))
            return {0.0, 0.0, 1};
        const double os = direct * direct / denom;
        return {os, os - direct * exchange / denom, 0};
    }
};

using Terms =
    thrust::transform_iterator<EnergyTerm, thrust::counting_iterator<int>>;

// These kernels have only O(1) scalar work, hence one thread. All O(N^5)
// transform/reduction work is cooperative cuBLAS/CUB work, not a serial kernel.
__global__ void initialize_packet(Packet* packet, const SystemScalars* scalars,
                                  bool accumulate) {
    *packet = Packet{};
    packet->prior_ev = accumulate ? scalars->energy : 0.0;
}

__global__ void accumulate_channels(Packet* packet, const Channels* channels) {
    packet->channels = AddChannels{}(packet->channels, *channels);
}

// Setup-only pointer map: one lane owns one sigma slab; a warp writes
// contiguous pointer entries and the grid covers n slabs with a tail guard.
// There is no reduction, shared scratch, atomic, or scientific-data copy.
// Repeating the read-only Co pointer lets pointer-batched cuBLAS share Co
// while every output matrix remains disjoint.
__global__ void initialize_batches(const double** left, const double** coeff,
                                   double** output, double* second,
                                   double* first, const double* co, int n,
                                   int nv, int no) {
    const int sigma = blockIdx.x * blockDim.x + threadIdx.x;
    if (sigma >= n)
        return;
    left[sigma] = second + static_cast<std::size_t>(sigma) * nv * n;
    coeff[sigma] = co;
    output[sigma] = first + static_cast<std::size_t>(sigma) * nv * no;
}

__global__ void publish_energy(Packet* packet, SystemScalars* scalars,
                               double os_scale, double ss_scale,
                               unsigned* status) {
    packet->reference = scalars->energy / units::kHartreeInEv;
    packet->correlation =
        os_scale * packet->channels.os + ss_scale * packet->channels.ss;
    packet->total = packet->reference + packet->correlation;
    if (!isfinite(packet->reference) || !isfinite(packet->correlation)
        || !isfinite(packet->total))
        packet->channels.invalid |= 2;
    if (!packet->channels.invalid)
        scalars->energy =
            packet->prior_ev + packet->total * units::kHartreeInEv;
    *status = packet->channels.invalid;
}

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::length_error("cuda::scf::MP2: workspace size overflow");
    return a * b;
}

void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuda::scf::MP2: cuBLAS failed (status "
                                 + std::to_string(static_cast<int>(status))
                                 + ")");
}

} // namespace

MP2Device::~MP2Device() { clear_(); }

void MP2Device::clear_() noexcept {
    if (blas_)
        cublasDestroy(blas_);
    cudaFree(first_);
    cudaFree(second_);
    cudaFree(reduction_scratch_);
    cudaFree(channels_);
    cudaFree(packet_);
    cudaFree(batch_pointers_);
    if (h_status_)
        cudaFreeHost(h_status_);
    blas_ = nullptr;
    first_ = second_ = nullptr;
    reduction_scratch_ = channels_ = packet_ = nullptr;
    batch_pointers_ = nullptr;
    h_status_ = d_status_ = nullptr;
    reduction_bytes_ = 0;
    n_ = n_occ_ = frozen_ = 0;
}

void MP2Device::build(const SCFWorkspace& ws, int frozen, double os_scale,
                      double ss_scale, std::size_t budget, cuda::Frame& frame,
                      bool accumulate, cudaStream_t stream) {
    if (ws.n <= 0 || ws.n_occ < 0 || ws.n_occ > ws.n || frozen < 0
        || frozen > ws.n_occ || !ws.C || !ws.eig || !ws.ERI || !frame.scalars)
        throw std::invalid_argument(
            "cuda::scf::MP2: require device C/energies/ERI/Frame scalars and "
            "0 <= frozen <= n_occ <= n_AO");
    if (!std::isfinite(os_scale) || !std::isfinite(ss_scale))
        throw std::invalid_argument(
            "cuda::scf::MP2: spin scales must be finite");

    clear_();
    n_ = ws.n;
    n_occ_ = ws.n_occ;
    frozen_ = frozen;
    os_scale_ = os_scale;
    ss_scale_ = ss_scale;
    const int nv = n_ - n_occ_;
    const int no = n_occ_ - frozen_;
    std::size_t first_count = 0, second_count = 0;
    if (nv && no) {
        const std::size_t n2 = multiply(n_, n_);
        first_count = multiply(n2, n_);
        second_count = multiply(n2, nv);
        if (first_count
            > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::length_error(
                "cuda::scf::MP2: AO cube exceeds cuBLAS int dimensions");
        Terms terms(thrust::counting_iterator<int>(0),
                    EnergyTerm{nullptr, nullptr, 0, n_occ_, frozen_, no, nv});
        ATV_CUDA_CHECK(cub::DeviceReduce::Reduce(
            nullptr, reduction_bytes_, terms, static_cast<Channels*>(nullptr),
            nv * no * nv, AddChannels{}, Channels{}, stream));
    }
    const std::size_t first_bytes = multiply(first_count, sizeof(double));
    const std::size_t second_bytes = multiply(second_count, sizeof(double));
    const std::size_t pointer_bytes =
        first_count ? multiply(multiply(n_, 3), sizeof(double*)) : 0;
    // Pointer arrays are bounded by the already-checked n^3 int dimension.
    const std::size_t fixed_bytes =
        sizeof(Packet) + sizeof(Channels) + sizeof(unsigned) + pointer_bytes;
    // Use subtraction to enforce the budget without overflowing an addition.
    if (fixed_bytes > budget || reduction_bytes_ > budget - fixed_bytes
        || first_bytes > budget - fixed_bytes - reduction_bytes_
        || second_bytes > budget - fixed_bytes - reduction_bytes_ - first_bytes)
        throw std::length_error("cuda::scf::MP2: occupied-tiled scratch "
                                "exceeds workspace_budget_bytes");
    const std::size_t required =
        fixed_bytes + reduction_bytes_ + first_bytes + second_bytes;
    std::size_t free_bytes = 0, total_bytes = 0;
    ATV_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (required > free_bytes - free_bytes / 5)
        throw std::length_error(
            "cuda::scf::MP2: insufficient free device memory for scratch");

    try {
        ATV_CUDA_CHECK(cudaMalloc(&packet_, sizeof(Packet)));
        ATV_CUDA_CHECK(cudaMalloc(&channels_, sizeof(Channels)));
        ATV_CUDA_CHECK(
            cudaHostAlloc(&h_status_, sizeof(unsigned), cudaHostAllocMapped));
        ATV_CUDA_CHECK(cudaHostGetDevicePointer(&d_status_, h_status_, 0));
        *h_status_ = 4u; // Pending: no scientific payload may be read yet.
        if (first_bytes) {
            ATV_CUDA_CHECK(cudaMalloc(&first_, first_bytes));
            ATV_CUDA_CHECK(cudaMalloc(&second_, second_bytes));
            ATV_CUDA_CHECK(cudaMalloc(&reduction_scratch_, reduction_bytes_));
            ATV_CUDA_CHECK(cudaMalloc(&batch_pointers_, pointer_bytes));
            check_blas(cublasCreate(&blas_));
            check_blas(cublasSetStream(blas_, stream));
            check_blas(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST));
            auto* bytes = static_cast<unsigned char*>(batch_pointers_);
            auto* left = reinterpret_cast<const double**>(bytes);
            auto* coeff =
                reinterpret_cast<const double**>(bytes + n_ * sizeof(double*));
            auto* output =
                reinterpret_cast<double**>(bytes + 2 * n_ * sizeof(double*));
            constexpr int block = 128;
            initialize_batches<<<(n_ + block - 1) / block, block, 0, stream>>>(
                left, coeff, output, second_, first_,
                ws.C + static_cast<std::size_t>(frozen_) * n_, n_, nv, no);
            ATV_CUDA_CHECK(cudaGetLastError());
        }
        initialize_packet<<<1, 1, 0, stream>>>(static_cast<Packet*>(packet_),
                                               frame.scalars, accumulate);
        ATV_CUDA_CHECK(cudaGetLastError());
    } catch (...) {
        clear_();
        throw;
    }
}

void MP2Device::evaluate(const SCFWorkspace& ws, cuda::Frame& frame,
                         cudaStream_t stream) {
    if (!packet_ || ws.n != n_ || ws.n_occ != n_occ_)
        throw std::logic_error(
            "cuda::scf::MP2: correlation workspace not prepared");
    const int nv = n_ - n_occ_;
    const int no = n_occ_ - frozen_;
    const int n2 = first_ ? n_ * n_ : 0;
    const int n3 = first_ ? n2 * n_ : 0;
    const double one = 1.0, zero = 0.0;
    for (int i = frozen_; nv && i < n_occ_; ++i) {
        // ERI is column-major [mu,nu,lambda,sigma]. Contract mu with C_i
        // without transposing the n^4 AO tensor: first[nu,lambda,sigma].
        check_blas(cublasDgemv(blas_, CUBLAS_OP_T, n_, n3, &one, ws.ERI, n_,
                               ws.C + static_cast<std::size_t>(i) * n_, 1,
                               &zero, first_, 1));
        // second[a,lambda,sigma] = Cv^T * first[nu,lambda,sigma].
        const double* cv = ws.C + static_cast<std::size_t>(n_occ_) * n_;
        check_blas(cublasDgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, nv, n2, n_,
                               &one, cv, n_, first_, n_, &zero, second_, nv));
        // first[a,j,sigma]. Each sigma slab shares the occupied C block.
        auto* bytes = static_cast<unsigned char*>(batch_pointers_);
        const auto* left = reinterpret_cast<const double* const*>(bytes);
        const auto* coeff = reinterpret_cast<const double* const*>(
            bytes + n_ * sizeof(double*));
        auto* output =
            reinterpret_cast<double**>(bytes + 2 * n_ * sizeof(double*));
        check_blas(cublasDgemmBatched(blas_, CUBLAS_OP_N, CUBLAS_OP_N, nv, no,
                                      n_, &one, left, nv, coeff, n_, &zero,
                                      output, nv, n_));
        // second[a,j,b], the only MO block materialized for this i tile.
        check_blas(cublasDgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, nv * no, nv, n_,
                               &one, first_, nv * no, cv, n_, &zero, second_,
                               nv * no));
        Terms terms(thrust::counting_iterator<int>(0),
                    EnergyTerm{second_, ws.eig, i, n_occ_, frozen_, no, nv});
        ATV_CUDA_CHECK(cub::DeviceReduce::Reduce(
            reduction_scratch_, reduction_bytes_, terms,
            static_cast<Channels*>(channels_), nv * no * nv, AddChannels{},
            Channels{}, stream));
        accumulate_channels<<<1, 1, 0, stream>>>(
            static_cast<Packet*>(packet_), static_cast<Channels*>(channels_));
        ATV_CUDA_CHECK(cudaGetLastError());
    }
    publish_energy<<<1, 1, 0, stream>>>(static_cast<Packet*>(packet_),
                                        frame.scalars, os_scale_, ss_scale_,
                                        d_status_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void MP2Device::validate(cudaStream_t stream) const {
    if (!packet_ || !h_status_)
        throw std::logic_error("cuda::scf::MP2: no prepared result storage");
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (*h_status_)
        throw std::invalid_argument("cuda::scf::MP2: non-negative/non-finite "
                                    "orbital denominator or non-finite result");
}

cpu::scf::MP2Result MP2Device::result(cudaStream_t stream,
                                      double& reference_energy) {
    validate(stream);
    Packet host{};
    ATV_CUDA_CHECK(cudaMemcpyAsync(&host, packet_, sizeof(host),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    reference_energy = host.reference;
    return {host.correlation, host.total,       false,
            frozen_,          host.channels.os, host.channels.ss};
}

} // namespace ATV::cuda::scf::internal

namespace ATV::cuda::scf {

MP2::MP2(RHF& mf, std::size_t budget)
    : workspace_budget_bytes(budget), mf_(&mf) {}

MP2::~MP2() = default;

MP2::MP2(MP2&& other) noexcept
    : frozen(other.frozen), os_scale(other.os_scale), ss_scale(other.ss_scale),
      workspace_budget_bytes(other.workspace_budget_bytes), mf_(other.mf_),
      device_(std::move(other.device_)), result_(other.result_) {
    other.mf_ = nullptr;
}

int MP2::default_budget() const { return mf_ ? mf_->max_iter : 0; }

void MP2::build_on(cuda::Context& ctx, cuda::Frame& frame) {
    if (!mf_)
        throw std::logic_error("cuda::scf::MP2: moved-from reference");
    mf_->build_on(ctx, frame);
}

void MP2::destroy_on(cuda::Context& ctx, cuda::Frame& frame) {
    device_.reset();
    if (mf_)
        mf_->destroy_on(ctx, frame);
}

int MP2::run_nested_erased(cuda::Context& ctx, cuda::Frame& frame, int budget) {
    if (!mf_)
        throw std::logic_error("cuda::scf::MP2: moved-from reference");
    mf_->energy_available_ = false;
    if (ctx.exec.forces_requested)
        throw std::invalid_argument(
            "cuda::scf::MP2: nuclear forces are not implemented; mount in an "
            "energy-only phase");
    try {
        return mf_->drive_loop_(ctx, frame, budget, false, false, this, true);
    } catch (...) {
        device_.reset();
        throw;
    }
}

void MP2::prepare_device_(const SCFWorkspace& ws, cuda::Context& ctx,
                          cuda::Frame& frame, bool accumulate) {
    device_ = std::make_unique<internal::MP2Device>();
    device_->build(ws, frozen, os_scale, ss_scale, workspace_budget_bytes,
                   frame, accumulate, static_cast<cudaStream_t>(ctx.stream));
}

void MP2::evaluate_device_(const SCFWorkspace& ws, cuda::Context& ctx,
                           cuda::Frame& frame, bool retrieve) {
    const auto stream = static_cast<cudaStream_t>(ctx.stream);
    device_->evaluate(ws, frame, stream);
    if (!retrieve) {
        // A nested evaluator publishes only to the parent's device Frame.
        // Check the control flag, never read a physical result mirror.
        device_->validate(stream);
        device_.reset();
    }
}

double MP2::retrieve_result_(cuda::Context& ctx) {
    double reference = 0.0;
    const auto candidate =
        device_->result(static_cast<cudaStream_t>(ctx.stream), reference);
    result_ = candidate;
    device_.reset();
    return reference;
}

cpu::scf::MP2Result MP2::solve_(cpu::Frame& frame, bool write) {
    if (!mf_)
        throw std::logic_error("cuda::scf::MP2: moved-from reference");
    mf_->energy_available_ = false;
    mf_->cache_host_types(frame);
    try {
        mf_->setup_pass(frame);
        auto& ctx = mf_->backend_->context();
        ctx.exec.forces_requested = false;
        ctx.exec.stress_requested = false;
        mf_->drive_loop_(ctx, mf_->backend_->frame(), mf_->max_iter, false,
                         true, this, false);
        if (write)
            mf_->teardown_pass(frame, false);
        else
            mf_->release_backend_();
    } catch (...) {
        const auto failure = std::current_exception();
        mf_->energy_available_ = false;
        device_.reset();
        try {
            mf_->release_backend_();
        } catch (...) {
            // Preserve the numerical/setup failure if cleanup also fails.
        }
        std::rethrow_exception(failure);
    }
    return result_;
}

cpu::scf::MP2Result MP2::run(cpu::Frame& frame) { return solve_(frame, true); }

cpu::scf::MP2Result MP2::run(const cpu::Frame& frame) {
    cpu::Frame scratch(frame);
    return solve_(scratch, false);
}

} // namespace ATV::cuda::scf

#endif // ATV_HAS_HF

/// @file cuda/dynamics/configurational_temperature.cu
/// @brief First-order configurational temperature on the device (see the
///        header).
#include "atomiverse/cuda/dynamics/configurational_temperature.cuh"

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

namespace {

constexpr int kRowFields = 10; // u(3), Hu(3), ∇²U, ∇Σ∇²U(3)
constexpr int kStats = 5;      // N, D, ΣN, ΣD, samples

// u = the decoded frame force; every accumulator row and the two sums
// start from zero.
__global__ void curvature_begin(const fixed_t* __restrict__ fx,
                                const fixed_t* __restrict__ fy,
                                const fixed_t* __restrict__ fz,
                                double* __restrict__ store, int stride,
                                double* __restrict__ sums, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
        sums[0] = sums[1] = 0.0;
    if (i >= n)
        return;
    store[i] = decode_force<double>(fx[i]);
    store[stride + i] = decode_force<double>(fy[i]);
    store[2 * stride + i] = decode_force<double>(fz[i]);
    for (int field = 3; field < kRowFields; ++field)
        store[static_cast<std::size_t>(field) * stride + i] = 0.0;
}

// N = Σ |u|², D = Σ ∇²U.
__global__ void curvature_reduce(CurvatureRows rows,
                                 double* __restrict__ sums) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    ::cuda::std::array<double, 2> values{};
    if (i < rows.n) {
        const double ux = rows.ux[i], uy = rows.uy[i], uz = rows.uz[i];
        values[0] = ux * ux + uy * uy + uz * uz;
        values[1] = rows.lap[i];
    }
    using Reduce =
        cub::BlockReduce<::cuda::std::array<double, 2>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out =
        Reduce(reduction).Reduce(values, SumComponents<double, 2>{});
    if (threadIdx.x == 0) {
        atomicAdd(sums, out[0]);
        atomicAdd(sums + 1, out[1]);
    }
}

// One sample into the observer's device statistics.
__global__ void accumulate_sample(const double* __restrict__ sums,
                                  double* __restrict__ stats) {
    stats[0] = sums[0];
    stats[1] = sums[1];
    stats[2] += sums[0];
    stats[3] += sums[1];
    stats[4] += 1.0;
}

} // namespace

CurvatureAssembly::CurvatureAssembly(CurvatureAssembly&& other) noexcept
    : sources_(std::move(other.sources_)), drives_(std::move(other.drives_)),
      store_(std::exchange(other.store_, nullptr)),
      sums_(std::exchange(other.sums_, nullptr)),
      rows_(std::exchange(other.rows_, CurvatureRows{})) {}

CurvatureAssembly&
CurvatureAssembly::operator=(CurvatureAssembly&& other) noexcept {
    if (this != &other) {
        destroy();
        sources_ = std::move(other.sources_);
        drives_ = std::move(other.drives_);
        store_ = std::exchange(other.store_, nullptr);
        sums_ = std::exchange(other.sums_, nullptr);
        rows_ = std::exchange(other.rows_, CurvatureRows{});
    }
    return *this;
}

CurvatureAssembly::~CurvatureAssembly() { destroy(); }

void CurvatureAssembly::set_terms(std::vector<CurvatureSource*> sources,
                                  std::vector<const ExternalDrive*> drives) {
    sources_ = std::move(sources);
    drives_ = std::move(drives);
}

void CurvatureAssembly::build(const BuildContext& ctx) {
    destroy();
    if (sources_.empty())
        throw std::runtime_error(
            "configurational temperature: the run has no CurvatureSource "
            "(no interaction potential)");
    if (ctx.frame.n_ghost() != 0)
        throw std::runtime_error(
            "configurational temperature: needs a single GPU (no ghost rows)");
    const std::size_t stride = static_cast<std::size_t>(ctx.frame.capacity());
    const std::size_t count = kRowFields * stride + 2;
    ATV_CUDA_CHECK(cudaMalloc(&store_, count * sizeof(double)));
    ATV_CUDA_CHECK(cudaMemset(store_, 0, count * sizeof(double)));
    sums_ = store_ + kRowFields * stride;
    double* const s = store_;
    rows_ = CurvatureRows{ctx.frame.n_local(), s,
                          s + stride,          s + 2 * stride,
                          s + 3 * stride,      s + 4 * stride,
                          s + 5 * stride,      s + 6 * stride,
                          s + 7 * stride,      s + 8 * stride,
                          s + 9 * stride};
}

void CurvatureAssembly::destroy() {
    cudaFree(store_);
    store_ = sums_ = nullptr;
    rows_ = CurvatureRows{};
}

void CurvatureAssembly::evaluate(const LaunchContext& ctx) const {
    const int n = ctx.frame.n_local();
    if (store_ == nullptr || n != rows_.n)
        throw std::runtime_error(
            "configurational temperature: evaluated before build, or the "
            "atom count changed since build");
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const auto& a = ctx.frame.atoms;
    const int stride = static_cast<int>(rows_.uy - rows_.ux);
    const int grid = grid_size(n);
    curvature_begin<<<grid, kDefaultBlock, 0, stream>>>(
        a.fx, a.fy, a.fz, store_, stride, sums_, n);
    for (const ExternalDrive* drive : drives_)
        drive->subtract_drive_force(ctx, n, store_, store_ + stride,
                                    store_ + 2 * stride);
    for (CurvatureSource* source : sources_)
        source->add_curvature(ctx, rows_);
    curvature_reduce<<<grid, kDefaultBlock, 0, stream>>>(rows_, sums_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

std::uint64_t CurvatureAssembly::epoch() const {
    std::uint64_t h = 0;
    for (const ExternalDrive* drive : drives_)
        h = h * 0x9E3779B97F4A7C15ull ^ drive->drive_epoch();
    return h;
}

ConfigurationalTemperature::ConfigurationalTemperature(int every)
    : every_(every) {
    if (every < 1)
        throw std::invalid_argument(
            "ConfigurationalTemperature: every must be >= 1");
}

ConfigurationalTemperature::~ConfigurationalTemperature() { cudaFree(stats_); }

ConfigurationalTemperature::ConfigurationalTemperature(
    ConfigurationalTemperature&& other) noexcept
    : every_(other.every_), assembly_(std::move(other.assembly_)),
      stats_(std::exchange(other.stats_, nullptr)), last_(other.last_),
      mean_(other.mean_), samples_(other.samples_) {}

ConfigurationalTemperature& ConfigurationalTemperature::operator=(
    ConfigurationalTemperature&& other) noexcept {
    if (this != &other) {
        cudaFree(stats_);
        every_ = other.every_;
        assembly_ = std::move(other.assembly_);
        stats_ = std::exchange(other.stats_, nullptr);
        last_ = other.last_;
        mean_ = other.mean_;
        samples_ = other.samples_;
    }
    return *this;
}

void ConfigurationalTemperature::set_curvature_terms(
    std::vector<CurvatureSource*> sources,
    std::vector<const ExternalDrive*> drives) {
    assembly_.set_terms(std::move(sources), std::move(drives));
}

void ConfigurationalTemperature::build(const BuildContext& ctx) {
    assembly_.build(ctx);
    cudaFree(stats_);
    stats_ = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&stats_, kStats * sizeof(double)));
    ATV_CUDA_CHECK(cudaMemset(stats_, 0, kStats * sizeof(double)));
    last_ = mean_ = ConfigurationalSample{};
    samples_ = 0;
}

void ConfigurationalTemperature::launch(const LaunchContext& ctx) {
    assembly_.evaluate(ctx);
    accumulate_sample<<<1, 1, 0,
                        static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
        assembly_.sums(), stats_);
    ATV_CUDA_CHECK(cudaGetLastError());
}

void ConfigurationalTemperature::setup_output(const LaunchContext& ctx) {
    ATV_CUDA_CHECK(
        cudaMemsetAsync(stats_, 0, kStats * sizeof(double),
                        static_cast<cudaStream_t>(ctx.runtime.stream)));
}

// The run has synchronized: read the statistics back (end-of-run only).
void ConfigurationalTemperature::on_retrieve(const BuildContext& ctx) {
    if (stats_ == nullptr)
        return;
    double host[kStats] = {};
    ATV_CUDA_CHECK(
        cudaMemcpy(host, stats_, sizeof(host), cudaMemcpyDeviceToHost));
    const double kB = ctx.runtime.units.boltz;
    samples_ = static_cast<long long>(host[4]);
    last_ = mean_ = ConfigurationalSample{};
    if (samples_ == 0)
        return;
    last_ = {host[0], host[1], host[0] / (kB * host[1])};
    const double count = host[4];
    mean_.force_sq = host[2] / count;
    mean_.laplacian = host[3] / count;
    mean_.temperature = mean_.force_sq / (kB * mean_.laplacian);
}

void ConfigurationalTemperature::destroy(const DestroyContext&) {
    assembly_.destroy();
    cudaFree(stats_);
    stats_ = nullptr;
}

} // namespace cuda
} // namespace ATV

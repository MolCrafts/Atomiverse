/// @file cuda/record/pressure_series.cu
/// @brief Freeze lab-frame kinetic stress and the updated cell, then write −P*.

#include "atomiverse/cuda/record/pressure_series.cuh"

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/record/pressure_series.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"

#include <fstream>
#include <stdexcept>

namespace ATV {
namespace cuda {
namespace md {
namespace {

__global__ void lab_kinetic_stress(const mixed_t* __restrict__ vx,
                                   const mixed_t* __restrict__ vy,
                                   const mixed_t* __restrict__ vz,
                                   const mixed_t* __restrict__ mass,
                                   double* __restrict__ kinetic, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    ::cuda::std::array<double, 6> mine{};
    if (i < n) {
        const double scale =
            static_cast<double>(mass[i]) * ATV::units::mass_v2_to_energy;
        const double vxi = static_cast<double>(vx[i]);
        const double vyi = static_cast<double>(vy[i]);
        const double vzi = static_cast<double>(vz[i]);
        mine[0] = scale * vxi * vxi;
        mine[1] = scale * vyi * vyi;
        mine[2] = scale * vzi * vzi;
        mine[3] = scale * vxi * vyi;
        mine[4] = scale * vxi * vzi;
        mine[5] = scale * vyi * vzi;
    }
    using Reduce =
        cub::BlockReduce<::cuda::std::array<double, 6>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out = Reduce(reduction).Reduce(mine, SumComponents<double, 6>{});
    if (threadIdx.x == 0) {
#pragma unroll
        for (int c = 0; c < 6; ++c)
            atomicAdd(&kinetic[c], out[c]);
    }
}

} // namespace

static_assert(3 == ATV::cuda::AsyncBridge::kRingDepth,
              "pressure series ring must match AsyncBridge");

PressureSeries::PressureSeries(std::string path, int every, double epsilon_ev,
                               int direction)
    : path_(std::move(path)), every_(every), epsilon_ev_(epsilon_ev),
      direction_(direction) {
    if (path_.empty())
        throw std::invalid_argument("PressureSeries: path is empty");
    if (every_ < 1)
        throw std::invalid_argument("PressureSeries: every must be >= 1");
    if (!(epsilon_ev_ > 0.0))
        throw std::invalid_argument("PressureSeries: epsilon must be positive");
    if (direction_ < 0 || direction_ > 5)
        throw std::invalid_argument("PressureSeries: direction must be 0..5");
}

void PressureSeries::on_build(const BuildContext& ctx) {
    if (ctx.frame.n_local() < 1 || ctx.frame.box == nullptr
        || ctx.frame.scalars == nullptr)
        throw std::invalid_argument(
            "PressureSeries: frame has no atoms or box");
    ATV_CUDA_CHECK(cudaMemcpy(cell0_, ctx.frame.box, 9 * sizeof(double),
                              cudaMemcpyDeviceToHost));
    for (int slot = 0; slot < 3; ++slot) {
        ATV_CUDA_CHECK(cudaMalloc(&device_[slot], sizeof(Payload)));
        ATV_CUDA_CHECK(cudaMallocHost(&host_[slot], sizeof(Payload)));
    }
    file_.open(path_);
    if (!file_.is_open())
        throw std::invalid_argument("PressureSeries: cannot open " + path_);
    file_ << cpu::kPressureSeriesHeader;
    file_.flush();
}

bool PressureSeries::can_run(const LaunchContext& ctx) const {
    const long long completed = static_cast<long long>(ctx.step) + 1;
    return completed % every_ == 0;
}

void PressureSeries::freeze_slot(const LaunchContext& ctx, int slot) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = frame.n_local();
    ATV_CUDA_CHECK(cudaMemsetAsync(device_[slot], 0, sizeof(Payload), stream));
    const int grid = cuda::grid_size(n);
    lab_kinetic_stress<<<grid, kDefaultBlock, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass,
        device_[slot]->kinetic, n);
    ATV_CUDA_CHECK(cudaMemcpyAsync(device_[slot]->stress, frame.scalars->stress,
                                   6 * sizeof(double), cudaMemcpyDeviceToDevice,
                                   stream));
    ATV_CUDA_CHECK(cudaMemcpyAsync(device_[slot]->cell, frame.box,
                                   9 * sizeof(double), cudaMemcpyDeviceToDevice,
                                   stream));
}

void PressureSeries::d2h_slot(const LaunchContext& ctx, int slot) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
    ATV_CUDA_CHECK(cudaMemcpyAsync(host_[slot], device_[slot], sizeof(Payload),
                                   cudaMemcpyDeviceToHost, stream));
}

void PressureSeries::on_data_ready(const LaunchContext& ctx, int slot) {
    cpu::PressureSeriesRow row;
    std::string error;
    const long long completed = static_cast<long long>(ctx.step) + 1;
    const Payload& sample = *host_[slot];
    if (!cpu::pressure_series_row(sample.stress, sample.kinetic, sample.cell,
                                  cell0_, direction_, epsilon_ev_, completed,
                                  row, error)) {
        file_ << "ERROR " << error << "\n";
        file_.flush();
        return;
    }
    cpu::write_pressure_series_row(file_, row);
    file_.flush();
}

void PressureSeries::on_destroy(const DestroyContext&) {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
    for (int slot = 0; slot < 3; ++slot) {
        if (device_[slot]) {
            cudaFree(device_[slot]);
            device_[slot] = nullptr;
        }
        if (host_[slot]) {
            cudaFreeHost(host_[slot]);
            host_[slot] = nullptr;
        }
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV

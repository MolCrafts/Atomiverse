/// @file cuda/record/pressure_series.cu
/// @brief −P* rows from captured velocities, masses, virial and cell.

#include "atomiverse/cuda/record/pressure_series.cuh"

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/record/pressure_series.h"
#include "atomiverse/cuda/device/error.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <stdexcept>

namespace ATV {
namespace cuda {
namespace md {

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
    mvv2e_ = ctx.runtime.units.mvv2e;
    if (ctx.frame.n_local() < 1 || ctx.frame.box == nullptr
        || ctx.frame.scalars == nullptr)
        throw std::invalid_argument(
            "PressureSeries: frame has no atoms or box");
    // The reference cell of the strain (build time may synchronize).
    ATV_CUDA_CHECK(cudaMemcpy(cell0_, ctx.frame.box, 9 * sizeof(double),
                              cudaMemcpyDeviceToHost));
    file_.open(path_);
    if (!file_.is_open())
        throw std::invalid_argument("PressureSeries: cannot open " + path_);
    file_ << cpu::kPressureSeriesHeader;
    file_.flush();
}

void PressureSeries::consume(const CaptureView& v) {
    // Lab-frame kinetic tensor Σ m v⊗v [eV] of the captured state.
    double kinetic[6] = {};
    for (int i = 0; i < v.n_atoms; ++i) {
        const double m = v.mass[i] * mvv2e_;
        const double vx = v.vx[i], vy = v.vy[i], vz = v.vz[i];
        kinetic[0] += m * vx * vx;
        kinetic[1] += m * vy * vy;
        kinetic[2] += m * vz * vz;
        kinetic[3] += m * vx * vy;
        kinetic[4] += m * vx * vz;
        kinetic[5] += m * vy * vz;
    }
    cpu::PressureSeriesRow row;
    std::string error;
    if (!cpu::pressure_series_row(v.scalars->stress, kinetic, v.cell, cell0_,
                                  direction_, epsilon_ev_, v.step, row,
                                  error)) {
        file_ << "ERROR " << error << "\n";
        file_.flush();
        return;
    }
    cpu::write_pressure_series_row(file_, row);
    file_.flush();
    if (!file_)
        throw std::runtime_error("cannot write the row");
}

void PressureSeries::on_destroy(const DestroyContext&) {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV

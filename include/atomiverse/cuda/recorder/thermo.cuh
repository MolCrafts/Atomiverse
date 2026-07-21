/// @file cuda/recorder/thermo.cuh
/// @brief Thermo -- header-only thermodynamic output recorder (async D2H bridge).
///
/// @see cpu::thermo_print (cpu/recorder/thermo.h)
#pragma once

#include <fstream>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "atomiverse/cpu/recorder/thermo.h"
#include "atomiverse/cuda/async/async_bridge.cuh"

namespace ATV {
namespace cuda {

/// @brief Thermodynamic output (LAMMPS thermo_style one).
///
/// Freezes potential + kinetic energy into a device-ring slot on the compute
/// stream each sampled step, so the value printed for step N is the step-N
/// value and is not corrupted by the next step's `ZeroEnergy` (a cross-stream
/// race the previous live-pointer D2H was exposed to).
///
/// @code{.cpp}
/// sim.on(md::Phase::end_of_step).add(Thermo{100});               // stdout only
/// sim.on(md::Phase::end_of_step).add(Thermo{100, "thermo.dat"}); // stdout + file
/// @endcode
class Thermo : public AsyncBridge {
  public:
    explicit Thermo(int every = 1, std::string filename = "")
        : every_(every), filename_(std::move(filename)) {}

    std::string name() const override { return "thermo"; }
    /// @brief Recorder — writes thermo line to stdout (and optional file).
    /* role via Recorder base */
    int frequency() const override { return every_; }

  protected:
    void on_build(const BuildContext& ctx) override {
        n_atoms_ = ctx.frame.n_atoms();
        header_printed_ = false;
        for (int i = 0; i < kRingDepth; ++i) {
            cudaMalloc(&d_pe_[i], sizeof(double)); // device-ring slot
            cudaMalloc(&d_ke_[i], sizeof(double));
            cudaMallocHost(&h_pe_[i], sizeof(double)); // pinned host slot
            cudaMallocHost(&h_ke_[i], sizeof(double));
            *h_pe_[i] = double(0);
            *h_ke_[i] = double(0);
        }
        if (!filename_.empty()) {
            file_.open(filename_);
            if (file_.is_open())
                file_ << "# Step Temp PotEng KinEng TotEng\n";
        }
    }

    void freeze_slot(const LaunchContext& ctx, int slot) override {
        // D2D freeze of the step-N scalars on the compute stream, before the
        // next step's pre_force ZeroEnergy can clear them.
        auto ms = static_cast<cudaStream_t>(ctx.runtime.stream);
        cudaMemcpyAsync(d_pe_[slot], &ctx.frame.scalars->energy, sizeof(double),
                        cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(d_ke_[slot], &ctx.frame.scalars->kinetic,
                        sizeof(double), cudaMemcpyDeviceToDevice, ms);
    }

    void d2h_slot(const LaunchContext& ctx, int slot) override {
        auto ds = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        cudaMemcpyAsync(h_pe_[slot], d_pe_[slot], sizeof(double),
                        cudaMemcpyDeviceToHost, ds);
        cudaMemcpyAsync(h_ke_[slot], d_ke_[slot], sizeof(double),
                        cudaMemcpyDeviceToHost, ds);
    }

    void on_data_ready(const LaunchContext& ctx, int slot) override {
        // CUDA has no constraint components yet, so Ndof is still 3N - 3 here.
        cpu::thermo_print(ctx.step, *h_pe_[slot], *h_ke_[slot],
                          3 * n_atoms_ - 3, header_printed_, file_);
    }

    void on_destroy(const DestroyContext&) override {
        if (file_.is_open())
            file_.close();
        for (int i = 0; i < kRingDepth; ++i) {
            if (d_pe_[i]) {
                cudaFree(d_pe_[i]);
                d_pe_[i] = nullptr;
            }
            if (d_ke_[i]) {
                cudaFree(d_ke_[i]);
                d_ke_[i] = nullptr;
            }
            if (h_pe_[i]) {
                cudaFreeHost(h_pe_[i]);
                h_pe_[i] = nullptr;
            }
            if (h_ke_[i]) {
                cudaFreeHost(h_ke_[i]);
                h_ke_[i] = nullptr;
            }
        }
    }

  private:
    int every_;
    std::string filename_;
    int n_atoms_ = 0;
    bool header_printed_ = false;
    std::ofstream file_;
    double* d_pe_[kRingDepth] =
        {}; ///< device-ring frozen potential energy [eV]
    double* d_ke_[kRingDepth] = {}; ///< device-ring frozen kinetic energy [eV]
    double* h_pe_[kRingDepth] = {}; ///< pinned host potential energy [eV]
    double* h_ke_[kRingDepth] = {}; ///< pinned host kinetic energy [eV]
};

} // namespace cuda
} // namespace ATV

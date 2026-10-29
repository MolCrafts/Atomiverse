/// @file cuda/record/thermo.cuh
/// @brief Thermo -- header-only thermodynamic output recorder (async D2H bridge).
///
/// @see ATV::ThermoStyle for the shared LAMMPS keyword set and format
#pragma once

#include <cmath>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/core/log.h"
#include "atomiverse/core/system_scalars.h"
#include "atomiverse/core/thermo_style.h"
#include "atomiverse/cuda/asynchronous/async_bridge.cuh"

namespace ATV {
namespace cuda {

/// @brief Thermodynamic output (LAMMPS `thermo N` + `thermo_style custom`).
///
/// Freezes the step-N system scalars and cell into a device-ring slot on the
/// compute stream each sampled step, so the line printed for step N is the
/// step-N state and is not corrupted by the next step's zeroing. The keywords
/// decide what the force field evaluates on those steps (`demand()`): `pe`,
/// `etotal` and `econserve` need the energy, the pressure keywords the
/// virial. Like LAMMPS, the last step of every run is printed too.
///
/// @code{.cpp}
/// sim.on(md::Phase::end_of_step).add(Thermo{100});                // default
/// sim.on(md::Phase::end_of_step).add(Thermo{100, {"step", "temp", "press"}});
/// sim.on(md::Phase::end_of_step).add(Thermo{100, {}, "thermo.dat"}); // + file
/// @endcode
class Thermo : public AsyncRecorder {
  public:
    /// @param every Print cadence [steps]; 0 prints every step.
    /// @param keywords `thermo_style custom` keywords; empty selects the
    ///        LAMMPS default `step temp pe ke etotal`.
    /// @param filename Optional second sink for the same lines.
    explicit Thermo(int every = 1, std::vector<std::string> keywords = {},
                    std::string filename = "")
        : every_(every),
          style_(keywords.empty() ? ThermoStyle{} : ThermoStyle{keywords}),
          filename_(std::move(filename)) {}

    std::string name() const override { return "thermo"; }
    /* role via AsyncRecorder / Recorder */
    int frequency() const override { return every_; }
    bool samples_final_step() const override { return true; }
    EvaluationDemand demand() const override {
        return {style_.needs_energy(), style_.needs_virial(),
                every_ > 0 ? every_ : 1};
    }

  protected:
    void on_build(const BuildContext& ctx) override {
        header_printed_ = false;
        for (int i = 0; i < kRingDepth; ++i) {
            cudaMalloc(&d_ring_[i], sizeof(Snapshot));     // device-ring slot
            cudaMallocHost(&h_ring_[i], sizeof(Snapshot)); // pinned host slot
            *h_ring_[i] = Snapshot{};
        }
        if (!filename_.empty()) {
            file_.open(filename_);
            if (file_.is_open())
                file_ << '#' << style_.header() << '\n';
        }
    }

    void freeze_slot(const LaunchContext& ctx, int slot) override {
        // D2D freeze of the step-N scalars and cell on the compute stream,
        // before the next step's zeroing can clear them.
        auto ms = static_cast<cudaStream_t>(ctx.runtime.stream);
        cudaMemcpyAsync(&d_ring_[slot]->scalars, ctx.frame.scalars,
                        sizeof(SystemScalars), cudaMemcpyDeviceToDevice, ms);
        cudaMemcpyAsync(d_ring_[slot]->cell, ctx.frame.box,
                        sizeof(Snapshot::cell), cudaMemcpyDeviceToDevice, ms);
    }

    void d2h_slot(const LaunchContext& ctx, int slot) override {
        auto ds = static_cast<cudaStream_t>(ctx.runtime.defer_stream);
        cudaMemcpyAsync(h_ring_[slot], d_ring_[slot], sizeof(Snapshot),
                        cudaMemcpyDeviceToHost, ds);
    }

    void on_data_ready(const LaunchContext& ctx, int slot) override {
        const Snapshot& snap = *h_ring_[slot];
        const SystemScalars& sc = snap.scalars;
        const auto* c = snap.cell;
        ThermoSample sample;
        sample.step = ctx.step;
        sample.pe = sc.energy;
        sample.ke = sc.kinetic;
        sample.conserved = sc.conserved;
        sample.volume = std::fabs(c[0] * (c[4] * c[8] - c[5] * c[7])
                                  - c[1] * (c[3] * c[8] - c[5] * c[6])
                                  + c[2] * (c[3] * c[7] - c[4] * c[6]));
        for (int a = 0; a < 6; ++a) {
            sample.virial[a] = sc.stress[a];
            sample.kinetic[a] = sc.stress_kinetic[a];
        }
        sample.ndof = ctx.runtime.md.n_dof_global;
        if (!header_printed_) {
            ATV_INFO("{}", style_.header());
            header_printed_ = true;
        }
        const std::string line = style_.line(sample);
        ATV_INFO("{}", line);
        if (file_.is_open())
            file_ << line << '\n';
    }

    void on_destroy(const DestroyContext&) override {
        if (file_.is_open())
            file_.close();
        for (int i = 0; i < kRingDepth; ++i) {
            if (d_ring_[i]) {
                cudaFree(d_ring_[i]);
                d_ring_[i] = nullptr;
            }
            if (h_ring_[i]) {
                cudaFreeHost(h_ring_[i]);
                h_ring_[i] = nullptr;
            }
        }
    }

  private:
    /// Frozen step state: scalars [eV] and the cell [Å, row-major].
    struct Snapshot {
        SystemScalars scalars;
        geom_t cell[9];
    };
    int every_;
    ThermoStyle style_;
    std::string filename_;
    bool header_printed_ = false;
    std::ofstream file_;
    Snapshot* d_ring_[kRingDepth] = {}; ///< Device ring.
    Snapshot* h_ring_[kRingDepth] = {}; ///< Pinned host ring.
};

} // namespace cuda
} // namespace ATV

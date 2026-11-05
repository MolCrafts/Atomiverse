/// @file cuda/record/thermo.cuh
/// @brief Thermo -- header-only thermodynamic output recorder (an
///        asynchronous consumer of the shared capture buffer).
///
/// @see ATV::ThermoStyle for the shared LAMMPS keyword set and format
#pragma once

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/log.h"
#include "atomiverse/core/system_scalars.h"
#include "atomiverse/core/thermo_style.h"
#include "atomiverse/cuda/asynchronous/async_consumer.cuh"

namespace ATV {
namespace cuda {

/// @brief Thermodynamic output (LAMMPS `thermo N` + `thermo_style custom`).
///
/// Reads the system scalars and the cell of each captured step from the
/// driver's shared capture buffer (`cuda::CaptureBuffer`), frozen on the
/// compute stream at the end of that step, so the line printed for step N is
/// the step-N state. The keywords decide what the force field evaluates on
/// those steps (`demand()`): `pe`, `etotal` and `econserve` need the energy,
/// the pressure keywords the virial. Like LAMMPS, the last step of every run
/// is printed too.
///
/// **Step convention (LAMMPS `thermo`).** The `step` column is the number of
/// completed MD steps and the line reports the state after exactly that many
/// steps (LAMMPS `update->ntimestep`). At the start of every run
/// `setup_output` prints the header and the current state (LAMMPS
/// `Output::setup`: step 0 on a fresh run; a continued run repeats the
/// previous run's last step under a new header, as LAMMPS does). The loop
/// prints after completed steps every, 2·every, … and on the run's last
/// step, so `run(frame, 1000)` at every 100 prints steps 0, 100, …, 1000.
///
/// @code{.cpp}
/// sim.on(md::Phase::end_of_step).add(Thermo{100});                // default
/// sim.on(md::Phase::end_of_step).add(Thermo{100, {"step", "temp", "press"}});
/// sim.on(md::Phase::end_of_step).add(Thermo{100, {}, "thermo.dat"}); // + file
/// @endcode
class Thermo : public AsyncRecorder {
  public:
    /// @param every Print cadence [steps]; 0 prints the first and last step
    ///        of each run only (LAMMPS `thermo 0`).
    /// @param keywords `thermo_style custom` keywords; empty selects the
    ///        LAMMPS default `step temp pe ke etotal`.
    /// @param filename Optional second sink for the same lines.
    explicit Thermo(int every = 1, std::vector<std::string> keywords = {},
                    std::string filename = "")
        : every_(every),
          style_(keywords.empty() ? ThermoStyle{} : ThermoStyle{keywords}),
          filename_(std::move(filename)) {}

    Thermo(Thermo&&) noexcept = default;
    Thermo& operator=(Thermo&&) noexcept = default;

    std::string name() const override { return "thermo"; }
    /// @brief Fires after completed steps every, 2·every, …; with every 0
    ///        never on cadence (a phase no step matches), so only the setup
    ///        line and the run's last step print (LAMMPS `thermo 0`).
    int frequency() const override { return every_ > 0 ? every_ : 1; }
    int frequency_phase() const override {
        return every_ > 0 ? every_ - 1 : -1;
    }
    bool samples_final_step() const override { return true; }
    /// @brief Energy / virial on the printed steps; with every 0 on the
    ///        reported final step (and setup) only.
    EvaluationDemand demand() const override {
        return {style_.needs_energy(), style_.needs_virial(),
                every_ > 0 ? every_ : 0, every_ > 0 ? every_ - 1 : 0};
    }
    /// @brief The scalars and the cell only (the slot header).
    CaptureColumns capture_columns() const override { return {}; }

    /// @brief Start of every run: header + the current state's line,
    ///        unconditionally (LAMMPS `Output::setup` calls
    ///        `thermo->header(); thermo->compute(0)` on every run).
    void setup_output(const LaunchContext& ctx) override {
        capture_now(ctx, ctx.step, CaptureKind::setup);
    }

  protected:
    void on_build(const BuildContext& ctx) override {
        runtime_ = &ctx.runtime;
        header_printed_ = false;
        if (!filename_.empty()) {
            file_.open(filename_);
            if (!file_.is_open())
                throw std::invalid_argument("cuda::Thermo: cannot open '"
                                            + filename_ + "'");
            file_ << '#' << style_.header() << '\n';
        }
    }

    void consume(const CaptureView& v) override {
        const SystemScalars& sc = *v.scalars;
        const double* c = v.cell;
        ThermoSample sample;
        sample.step = v.step;
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
        sample.ndof = runtime_->md.n_dof_global;
        // The setup line opens a run (LAMMPS prints the header per run).
        if (v.kind == CaptureKind::setup || !header_printed_) {
            ATV_INFO("{}", style_.header());
            header_printed_ = true;
        }
        const std::string line = style_.line(sample);
        ATV_INFO("{}", line);
        if (file_.is_open()) {
            file_ << line << '\n';
            if (!file_)
                throw std::runtime_error("cuda::Thermo: writing '" + filename_
                                         + "' failed");
        }
    }

    void on_destroy(const DestroyContext&) override {
        if (file_.is_open())
            file_.close();
    }

  private:
    int every_;
    ThermoStyle style_;
    std::string filename_;
    const cuda::Context* runtime_ = nullptr; ///< n_dof_global per line
    bool header_printed_ = false;
    std::ofstream file_;
};

} // namespace cuda
} // namespace ATV

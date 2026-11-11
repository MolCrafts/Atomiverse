/// @file cpu/record/thermo.h
/// @brief CPU thermodynamic output component (LAMMPS `thermo` +
///        `thermo_style custom`).
///
/// Keywords and line format are shared with `cuda::Thermo` through
/// `ATV::ThermoStyle`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/thermo_style.h"
#include "atomiverse/cpu/component/recorder.h"

namespace ATV {
namespace cpu {

/// @brief Thermodynamic output recorder component (CPU backend).
///
/// Prints the `thermo_style custom` columns of the frame state. MPI-aware:
/// only rank 0 prints.
///
/// **Step convention (LAMMPS `thermo`).** The `step` column is the number
/// of completed MD steps and the line reports the state after exactly that
/// many steps (LAMMPS `update->ntimestep`). At the start of every
/// `cpu::md::MD` run `setup_output` prints the header and the current state
/// (LAMMPS `Output::setup`: step 0 on a fresh run; a continued run repeats
/// the previous run's last step under a new header, as LAMMPS does). The
/// end_of_step launch of the step with zero-based index `ctx.step` prints
/// step `ctx.step + 1` when that is a multiple of `freq`, and always on the
/// last step of a run (LAMMPS `Output::write`). So `run(frame, 1000)` at
/// freq 100 prints steps 0, 100, …, 1000, and at freq 0 steps 0 and 1000
/// only (LAMMPS `thermo 0`). Under `cpu::minimizer::Minimize` a step is an
/// iteration, and a minimization stopped by a criterion before its budget
/// also prints its final iteration (LAMMPS `Min::run`); a single-pass
/// `cpu::Evaluate` (LAMMPS `run 0`) prints its one state whatever the
/// cadence. No step prints twice
/// within a run (LAMMPS `last_thermo`). The cadence is applied here, not by
/// the Runtime's `frequency()` gate, because the last-step line falls off
/// cadence.
class Thermo : public cpu::Recorder {
  public:
    /// @param freq Print cadence [steps]; 0 prints the first and last step
    ///        of each run only (LAMMPS `thermo 0`).
    /// @param keywords `thermo_style custom` keywords; empty selects the
    ///        LAMMPS default `step temp pe ke etotal`.
    explicit Thermo(int freq, std::vector<std::string> keywords = {});

    std::string name() const override { return "thermo"; }
    /* role via Recorder base */
    /// @brief Start of a run: header + the line of the current state.
    void setup_output(const cpu::LaunchContext& ctx) override;
    /// @brief End of a step: the line of step `ctx.step + 1` when on
    ///        cadence or the run's last step.
    void launch(const cpu::LaunchContext& ctx) override;
    /// @brief Minimization stopped early: the final iteration's line,
    ///        unless already printed.
    void final_output(const cpu::LaunchContext& ctx) override;

  private:
    int freq_;
    ThermoStyle style_;
    bool header_printed_ = false;
    BigInt last_step_ = -1; ///< LAMMPS `last_thermo`

    /// @brief Print (rank 0) the line of the current state as `step`
    ///        [completed steps], preceded by the header when asked.
    void print_(const cpu::LaunchContext& ctx, BigInt step, bool header);
};

} // namespace cpu
} // namespace ATV

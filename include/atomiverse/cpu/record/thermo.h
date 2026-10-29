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
/// Prints the `thermo_style custom` columns of the frame state at the given
/// frequency. MPI-aware: only rank 0 prints.
class Thermo : public cpu::Recorder {
  public:
    /// @param freq Print cadence [steps]; 0 prints every step.
    /// @param keywords `thermo_style custom` keywords; empty selects the
    ///        LAMMPS default `step temp pe ke etotal`.
    explicit Thermo(int freq, std::vector<std::string> keywords = {});

    std::string name() const override { return "thermo"; }
    /* role via Recorder base */
    int frequency() const override { return freq_; }
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    int freq_;
    ThermoStyle style_;
    bool header_printed_ = false;
};

} // namespace cpu
} // namespace ATV

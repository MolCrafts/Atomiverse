/// @file cpu/recorder/thermo.h
/// @brief CPU thermodynamic output: Thermo component + thermo_print free function.
///
/// `Thermo` is the per-step recorder component; `thermo_print` is the
/// low-level formatted-print helper called by `cuda::Thermo` after async D2H.
#pragma once

#include <fstream>
#include <string>

#include "atomiverse/cpu/component/recorder.h"

namespace ATV {
namespace cpu {

/// @brief Thermodynamic output recorder component (CPU backend).
///
/// Prints step, PE, KE, temperature, total energy, and the
/// integrator's conserved quantity (`frame.conserved` — the extended
/// Hamiltonian for Nosé-Hoover NVT/NPT) at the given frequency.
/// MPI-aware: only rank 0 prints.
class Thermo : public cpu::Recorder {
  public:
    explicit Thermo(int freq);

    std::string name() const override { return "thermo"; }
    /* role via Recorder base */
    int frequency() const override { return freq_; }
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    int freq_;
};

/// @brief Print one thermodynamic output line.
///
/// Prints a column header on the first call (guarded by header_printed).
/// Writes to stdout and, if file is open, to the optional log file as well.
///
/// @param step           Current step number.
/// @param pe             Potential energy [eV].
/// @param ke             Kinetic energy [eV].
/// @param n_atoms        Number of atoms (used to compute degrees of freedom).
/// @param header_printed Guard flag; set to true after the first call.
/// @param file           Optional open output stream (may be a closed fstream).
void thermo_print(int step, double pe, double ke, int n_dof,
                  bool& header_printed, std::ofstream& file);

} // namespace cpu
} // namespace ATV

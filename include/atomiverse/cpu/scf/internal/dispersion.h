/// @file cpu/scf/internal/dispersion.h
/// @brief The ONE place that answers "which dispersion correction did this
///        method fold into its energy, and what is its gradient".
///
/// `MeanFieldMethod` exposes two independent flags (`d3bj`, `d4`) and adds the
/// chosen E_disp to the total energy it reports. Every gradient the method
/// publishes must therefore carry −∂E_disp/∂R for the *same* choice: the
/// mean-field `forces()` (through `ScfGradient`), and the post-HF
/// `mp2_forces()` / `ccsd_forces()`, which build their energy on the same
/// `e_total_`. Those are three call sites for one decision, and the defect
/// this header exists to prevent — an energy term whose force was simply
/// missing — is exactly what happens when the decision is spelled out three
/// times and only two of them get updated.
///
/// The dispersion *math* is not here and is not duplicated anywhere: D3(BJ) is
/// `cpu/gpw/internal/dftd3.cpp` (shared verbatim with periodic GPW, analytic
/// two-body + dC6/dCN chain), D4 is `cpu/scf/dftd4.cpp`. This header owns only
/// the flag → evaluator switch and the unit bookkeeping.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief E_disp for the flavour the flags select.
///
/// `d4` wins when both are set (a method sets one; the precedence exists only
/// so that energy and force can never disagree about which one).
/// @param frame System geometry (Å, Z).
/// @param d3bj  PBE-D3(BJ) requested.
/// @param d4    PBE-D4 requested.
/// @return E_disp [Hartree]; exactly 0 when neither flag is set.
double dispersion_energy(const Frame& frame, bool d3bj, bool d4);

/// @brief \(F_A = -\partial E_{disp}/\partial R_A\) for the same flavour.
/// @param frame System geometry (Å, Z).
/// @param d3bj  PBE-D3(BJ) requested.
/// @param d4    PBE-D4 requested.
/// @return One xyz triple per atom [Hartree/Bohr]; **empty** when neither flag
///         is set (callers skip the leg rather than adding zeros).
std::vector<std::array<double, 3>> dispersion_forces(const Frame& frame,
                                                     bool d3bj, bool d4);

/// @brief Add the dispersion leg to a gradient already in the public currency.
///
/// For the post-HF paths, whose `GradientResult` is assembled in eV/Å by a
/// correlated-gradient routine that knows nothing about dispersion, while its
/// `energy` came from an `e_total_` that already includes E_disp.
/// @param frame System geometry (Å, Z).
/// @param d3bj  PBE-D3(BJ) requested.
/// @param d4    PBE-D4 requested.
/// @param gr    Gradient to update in place (forces [eV/Å]); untouched when
///              neither flag is set.
/// @throws std::runtime_error when the evaluator's atom count disagrees with
///         the gradient's.
void add_dispersion_forces(const Frame& frame, bool d3bj, bool d4,
                           GradientResult& gr);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

/// @file cpu/scf/internal/tda_gradient.h
/// @brief **PRIVATE** analytic closed-shell CIS/TDA nuclear gradient workspace.
///
/// Excited-state force F_n = −∇_R (E_HF + ω_n) in eV/Å using skeleton
/// integral derivatives + unrelaxed CIS difference / transition densities
/// (Foresman et al.; Maurice & Head-Gordon). Orbital response via one
/// Z-vector solve on the link-08 singlet (A+B) kernel.
#pragma once

#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Analytic CIS/TDA nuclear forces on root `state` [eV/Å], length 3N.
/// @param frame Geometry [Å] (read-only positions).
/// @param pack  Integral pack at this geometry (ERI may be rebuilt).
/// @param det   Converged closed-shell RHF determinant.
/// @param e_scf_ha SCF energy [Hartree] (for total-energy assembly only).
/// @param state 0-based TDA root index.
/// @return Force array fx0,fy0,fz0,... [eV/Å].
std::vector<double> tda_analytic_forces(const Frame& frame,
                                        const internal::IntegralView& pack,
                                        const SlaterDeterminant& det,
                                        double e_scf_ha, int state);

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV

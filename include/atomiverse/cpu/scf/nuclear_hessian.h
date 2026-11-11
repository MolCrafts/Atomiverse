/// @file cpu/scf/nuclear_hessian.h
/// @brief Nuclear second-derivative helpers that are not mean-field methods.
///
/// `pyscf.scf.cphf.solve` is a free function: the coupled-perturbed response
/// is not a method of `scf.hf.RHF`. `pyscf.hessian.rhf.hess_elec` is a
/// function of a Hessian object (`hessobj.base` is the mean field), not of
/// the SCF class. The finite-difference skeleton that RKS/UKS/ROHF share
/// follows that split: it differentiates a caller's forces and does not live
/// on `MeanFieldMethod`.
#pragma once

#include <functional>
#include <string_view>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Central finite-difference nuclear Hessian [eV/Å²] from analytic
///        forces.
///
/// Step h = 1e-4 Bohr (the offline PySCF pin). Column index is atom-major
/// `col = 3A+d`. The displacement loop is serial. Geometry is restored to
/// the entry coordinates. Symmetrizes `½(H+Hᵀ)`.
/// @param frame Geometry [Å]; mutated during the sweep and restored.
/// @param forces_at Analytic forces at the frame's current geometry [eV/Å].
/// @return Symmetric nuclear Hessian [eV/Å²].
vib::HessianResult central_difference_nuclear_hessian(
    Frame& frame, const std::function<GradientResult(Frame&)>& forces_at);

/// @brief Throw if a converged reference cannot support an analytic second
///        derivative (SCF not converged, spherical AOs, or an ECP).
/// @param method_name Label interpolated into the message, e.g. "Hessian::run".
/// @param converged Whether the reference SCF converged.
/// @param stop_reason Driver stop reason when it did not.
/// @param pack Integral view of the reference.
void check_hessian_preconditions(std::string_view method_name, bool converged,
                                 std::string_view stop_reason,
                                 const internal::IntegralView& pack);

} // namespace scf
} // namespace cpu
} // namespace ATV

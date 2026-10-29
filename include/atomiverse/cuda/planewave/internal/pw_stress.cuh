/// @file cuda/planewave/internal/pw_stress.cuh
/// @brief Frozen-density NSCF strain-FD stress from the method's one device SCF.
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include <array>

namespace ATV::cuda::pw {
class PW;
namespace internal {
/// @brief Twelve fixed-basis NSCF strain samples at the supplied device density.
/// No reference SCF is performed here. The reference problem retains Miller
/// labels/FFT shape; both spin spectra share one chemical potential under smear.
/// The existing density stream is borrowed by every strained device evaluation.
/// @param frame Reference geometry [Å].
/// @param method Functional, smearing and numerical identity of the reference.
/// @param reference Problem that owns the accepted finite variational space.
/// @param density Reference mixed density; it is never changed by the probes.
/// @param n_alpha Occupied alpha seed (-1 for restricted).
/// @param n_beta Occupied beta seed (-1 for restricted).
/// @return Intensive stress [eV/Å³], xx,yy,zz,xy,xz,yz (symmetric-shear lanes).
std::array<double, 6>
pw_stress_from_state(const cpu::Frame& frame, const PW& method,
                     const cpu::pw::internal::PWProblem& reference,
                     DeviceDensity& density, int n_alpha, int n_beta);
} // namespace internal
} // namespace ATV::cuda::pw

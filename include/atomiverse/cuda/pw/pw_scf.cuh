/// @file cuda/pw/pw_scf.cuh
/// @brief cuda::pw::pw_scf() — GPU-accelerated plane-wave DFT total energy.
///
/// Same contract and units as `cpu::pw::pw_scf` (the two share the host
/// problem assembly in cpu/pw/internal/pw_setup.h). The SCF loop runs the
/// band-structure hot path — Davidson H·ψ FFTs, KB projector products and
/// per-band density accumulation — on the GPU via cuFFT/cuBLAS, while the
/// cheap O(N_G) density-side steps (Poisson, LDA via the shared XC
/// implementation, V_eff assembly, Pulay mixing, convergence check) reuse
/// the CPU components verbatim for exact numerical parity.
///
/// Like the Gaussian-basis `cuda::scf::SCFSolver`, the plane-wave CUDA
/// solver is an inner solver: no Driver, no phase enum, no Plan
/// registration.
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/pw_scf.h"
#include <string>

namespace ATV {
namespace cuda {
namespace pw {

/// @brief Result type shared with the CPU driver (energy in Ry).
using PWResult = ::ATV::cpu::pw::PWResult;

/// @brief Run a plane-wave LDA total-energy SCF on the GPU.
///
/// @param frame      Input structure: positions, atom types (Z), cell.
/// @param upf_path   Path to UPF pseudopotential file (empty = toy V_loc).
/// @param ecut_ry    Plane-wave energy cutoff [Ry].
/// @param k_mesh     Monkhorst-Pack mesh size per dimension.
/// @param tol        SCF convergence tolerance on density change.
/// @param max_iter   Maximum SCF iterations.
/// @return PWResult with energy (Ry), converged, iterations.
PWResult pw_scf(const cpu::Frame& frame, const std::string& upf_path = "",
                double ecut_ry = 30.0, int k_mesh = 4, double tol = 1e-6,
                int max_iter = 50);

} // namespace pw
} // namespace cuda
} // namespace ATV

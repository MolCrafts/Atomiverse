/// @file cuda/pw/internal/xc_lda.cuh
/// @brief Real-space LDA exchange-correlation on the plane-wave grid — device
///        kernel mirroring cpu::pw::internal::XCBuilder's per-point XC evaluation.
#pragma once

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief Evaluate the LDA XC potential and energy density on the real grid.
///
/// Per grid point this reproduces cpu::pw::internal::XCBuilder exactly: with total density
/// `rho < 1e-12` both outputs are zero; otherwise it computes
/// `evaluate_xc(PBE, rho/2, rho/2, sigma=0)`. At zero density gradient the PBE
/// GGA enhancement is unity and its correlation gradient term vanishes, so this
/// reduces analytically to Slater exchange + PW92 LSDA correlation (the same
/// arithmetic the CPU dispatch executes). Spin-unpolarized (ζ=0).
///
/// @param d_rho_r  Real-space total density ρ(r), nr reals [electrons/Bohr³].
/// @param d_vxc_r  Output V_xc(r) = ∂(ρε_xc)/∂ρ_α, nr reals [Hartree].
/// @param d_exc_r  Output ε_xc(r) per electron, nr reals [Hartree].
/// @param nr       Number of real-space grid points.
/// @param stream   CUDA stream (default stream if null).
void xc_lda_grid(const double* d_rho_r, double* d_vxc_r, double* d_exc_r,
                 int nr, cudaStream_t stream = nullptr);

/// @brief Accumulate the XC energy E_xc = dV · Σ_i ρ(r)_i ε_xc(r)_i.
///
/// Mirrors the E_xc accumulation in cpu::pw::internal::XCBuilder's grid loop, where the
/// volume element is dV = Ω/nr. The reduction uses double-precision atomic
/// accumulation, so the summation order is not bit-reproducible run-to-run
/// (differences are at the ~1e-11 relative level, far below SCF tolerance).
///
/// @param d_rho_r Real-space density ρ(r), nr reals (device).
/// @param d_exc_r Per-electron XC energy density ε_xc(r), nr reals (device).
/// @param nr      Number of real-space grid points.
/// @param dV      Volume element Ω/nr [Bohr³].
/// @param d_Exc   Output scalar E_xc [Hartree], 1 double (device). Overwritten.
/// @param stream  CUDA stream (default stream if null).
void xc_energy(const double* d_rho_r, const double* d_exc_r, int nr, double dV,
               double* d_Exc, cudaStream_t stream = nullptr);

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV

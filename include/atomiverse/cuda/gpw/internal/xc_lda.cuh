/// @file cuda/gpw/internal/xc_lda.cuh
/// @brief Real-space GTH Padé LDA XC on the GPW uniform grid — device path.
///
/// Thin CUDA foothold mirroring `cpu::gpw::xc_grid_pade` / `cpu::xc::pade_lsd`
/// (CP2K-native PADE, not libxc). Per-point arithmetic is transcribed from
/// `src/cpu/xc/pade.cpp` so device results match the CPU oracle to ~1e-12.
/// Not a full GPW SCF product: density collocation, KS integrate, and the
/// outer Driver loop stay on CPU.
#pragma once

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace gpw {
namespace internal {

/// @brief Evaluate unpolarized Padé LDA V_xc and ε_xc on a real-space grid.
///
/// For each point with total density `rho`, evaluates
/// `cpu::xc::pade_lsd(rho/2, rho/2)`: when `rho_total < 1e-12` both outputs
/// are zero; otherwise `vxc = vrho_a` (= `vrho_b` for ζ=0) and
/// `exc` is the per-electron XC energy density. Spin-unpolarized (ζ=0).
///
/// @param d_rho_r Real-space total density ρ(r), nr reals [e/Bohr³].
/// @param d_vxc_r Output V_xc(r) = ∂(ρε_xc)/∂ρ_α, nr reals [Hartree].
/// @param d_exc_r Output ε_xc(r) per electron, nr reals [Hartree].
/// @param nr      Number of real-space grid points.
/// @param stream  CUDA stream (default stream if null).
void xc_lda_grid(const double* d_rho_r, double* d_vxc_r, double* d_exc_r,
                 int nr, cudaStream_t stream = nullptr);

/// @brief Spin-resolved Padé LSD on the real-space grid (UKS twin).
///
/// Per point: `cpu::xc::pade_lsd(rho_a, rho_b)` → vrho_a / vrho_b / exc.
/// Total density below 1e-12 zeroes all three channels.
///
/// @param d_rho_a Alpha density, nr reals [e/Bohr³].
/// @param d_rho_b Beta density, nr reals [e/Bohr³].
/// @param d_vxc_a Output V_xc^α, nr reals [Hartree].
/// @param d_vxc_b Output V_xc^β, nr reals [Hartree].
/// @param d_exc_r Output ε_xc per electron, nr reals [Hartree].
/// @param nr      Number of real-space grid points.
/// @param stream  CUDA stream (default stream if null).
void xc_lda_grid_lsd(const double* d_rho_a, const double* d_rho_b,
                     double* d_vxc_a, double* d_vxc_b, double* d_exc_r, int nr,
                     cudaStream_t stream = nullptr);

/// @brief Accumulate E_xc = w · Σ_i ρ_i ε_xc,i (spin-unpolarized).
///
/// Mirrors the closed-shell reduction in `cpu::gpw::xc_grid_pade`, where
/// `w = V_cell / npts` is the grid integration weight. Device atomic
/// accumulation is not bit-reproducible run-to-run (relative differences
/// ~1e-11, far below SCF tolerance).
///
/// @param d_rho_r Real-space total density ρ(r), nr reals (device).
/// @param d_exc_r Per-electron XC energy density ε_xc(r), nr reals (device).
/// @param nr      Number of real-space grid points.
/// @param w       Grid weight V_cell/npts [Bohr³].
/// @param d_Exc   Output scalar E_xc [Hartree], 1 double (device). Overwritten.
/// @param stream  CUDA stream (default stream if null).
void xc_energy(const double* d_rho_r, const double* d_exc_r, int nr, double w,
               double* d_Exc, cudaStream_t stream = nullptr);

/// @brief Accumulate E_xc = w · Σ_i (ρ_α+ρ_β)_i ε_xc,i (spin-resolved).
///
/// Mirrors `cpu::gpw::xc_grid_pade_lsd`.
///
/// @param d_rho_a Alpha density, nr reals (device).
/// @param d_rho_b Beta density, nr reals (device).
/// @param d_exc_r Per-electron XC energy density, nr reals (device).
/// @param nr      Number of real-space grid points.
/// @param w       Grid weight V_cell/npts [Bohr³].
/// @param d_Exc   Output scalar E_xc [Hartree], 1 double (device). Overwritten.
/// @param stream  CUDA stream (default stream if null).
void xc_energy_lsd(const double* d_rho_a, const double* d_rho_b,
                   const double* d_exc_r, int nr, double w, double* d_Exc,
                   cudaStream_t stream = nullptr);

} // namespace internal
} // namespace gpw
} // namespace cuda
} // namespace ATV

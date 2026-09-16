/// @file cuda/planewave/internal/xc_lda.cuh
/// @brief Real-space LSDA exchange-correlation on the plane-wave grid —
///        device kernel mirroring `cpu::pw::internal::XCBuilder::step(ws,
///        XCType::LSDA)`'s per-point XC evaluation.
#pragma once

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief Evaluate the LSDA XC potential and energy density on the real grid.
///
/// Per grid point this reproduces `cpu::scf::evaluate_xc(XCType::LSDA, ρ/2,
/// ρ/2)` — Slater exchange + VWN5 correlation — including the `ρ < 1e-12`
/// cutoff that zeroes both outputs. Spin-unpolarized (ζ = 0): the VWN5 spin
/// interpolation f(ζ) and its derivative both vanish identically at ζ = 0
/// (f(0) = f'(0) = 0, ζ⁴ = 0), so the ferromagnetic and spin-stiffness Padé
/// channels drop out exactly and only the paramagnetic set is needed —
/// ε_c = F_P(x), v_c = ε_c − (x/6) F_P'(x) with x = √r_s.
///
/// This is NOT the σ = 0 limit of PBE: that limit uses PW92 correlation, which
/// no public `XCType` value selects, and disagrees with VWN5 by ~1e-4 Ha at
/// typical valence densities.
///
/// @param d_rho_r  Real-space total density ρ(r), nr reals [electrons/Bohr³].
/// @param d_vxc_r  Output V_xc(r) = ∂(ρ ε_xc)/∂ρ_α, nr reals [Hartree].
/// @param d_exc_r  Output ε_xc(r) per electron, nr reals [Hartree].
/// @param nr       Number of real-space grid points.
/// @param stream   CUDA stream (default stream if null).
void xc_lsda_grid(const double* d_rho_r, double* d_vxc_r, double* d_exc_r,
                  int nr, cudaStream_t stream = nullptr);

/// @brief Collinear LSDA XC on spin densities ρ_α(r), ρ_β(r).
///
/// Mirrors `cpu::scf::evaluate_xc(XCType::LSDA, ρ_α, ρ_β)` — Slater exchange +
/// full VWN5 correlation with the spin-interpolation channel — including the
/// total-density `ρ_α+ρ_β < 1e-12` cutoff and the per-spin floor at 1e-15 on
/// the correlation inputs only (CPU parity).
///
/// @param d_rho_a  Real-space α density, nr reals [electrons/Bohr³].
/// @param d_rho_b  Real-space β density, nr reals [electrons/Bohr³].
/// @param d_vxc_a  Output V_xc^α(r), nr reals [Hartree].
/// @param d_vxc_b  Output V_xc^β(r), nr reals [Hartree].
/// @param d_exc_r  Output ε_xc(r) per electron, nr reals [Hartree].
/// @param nr       Number of real-space grid points.
/// @param stream   CUDA stream (default stream if null).
void xc_lsda_spin_grid(const double* d_rho_a, const double* d_rho_b,
                       double* d_vxc_a, double* d_vxc_b, double* d_exc_r,
                       int nr, cudaStream_t stream = nullptr);

/// @brief Stage-1 partial count of the `xc_energy` reduction — the required
///        length (in doubles) of the caller-owned scratch buffer.
inline constexpr int kXcEnergyPartials = 256;

/// @brief Accumulate the XC energy E_xc = dV · Σ_i ρ(r)_i ε_xc(r)_i.
///
/// Mirrors the E_xc accumulation in `cpu::pw::internal::XCBuilder`'s grid loop,
/// where the volume element is dV = Ω/nr. The reduction is a **fixed-grid**
/// two-stage tree (`kXcEnergyPartials` blocks × 256 threads, grid-stride), so
/// the summation partition — and therefore the result — is bitwise
/// reproducible run-to-run for a given `nr`. The previous `atomicAdd` form was
/// not. The CPU accumulates `ρ·ε_xc·dV` per point while this scales the total
/// once; both agree to rounding, and neither is bit-identical to the other
/// (serial vs tree summation, ~1e-16 relative).
///
/// @param d_rho_r  Real-space density ρ(r), nr reals (device).
/// @param d_exc_r  Per-electron XC energy density ε_xc(r), nr reals (device).
/// @param nr       Number of real-space grid points.
/// @param dV       Volume element Ω/nr [Bohr³].
/// @param d_Exc    Output scalar E_xc [Hartree], 1 double (device).
///                 Overwritten.
/// @param d_scratch Caller-owned staging buffer, `kXcEnergyPartials` doubles
///                 (device). Contents are clobbered. Explicit rather than a
///                 hidden `__device__` global so the launcher stays reentrant
///                 across streams and allocates nothing per call.
/// @param stream   CUDA stream (default stream if null).
void xc_energy(const double* d_rho_r, const double* d_exc_r, int nr, double dV,
               double* d_Exc, double* d_scratch, cudaStream_t stream = nullptr);

/// @brief Collinear E_xc = dV · Σ_i (ρ_α + ρ_β)_i ε_xc(r)_i.
///
/// Mirrors `XCBuilder::step_spin`'s E_xc accumulation (floored densities are
/// the caller's responsibility — same as the restricted path).
void xc_energy_spin(const double* d_rho_a, const double* d_rho_b,
                    const double* d_exc_r, int nr, double dV, double* d_Exc,
                    double* d_scratch, cudaStream_t stream = nullptr);

/// @brief Collinear double-counting ∫(ρ_α V_xc^α + ρ_β V_xc^β) dV.
///
/// Mirrors `ws.e_xc_int_spin` in `XCBuilder::step_spin` (real-space form used
/// by the spin energy decomposition).
void xc_int_spin(const double* d_rho_a, const double* d_rho_b,
                 const double* d_vxc_a, const double* d_vxc_b, int nr,
                 double dV, double* d_e_xc_int, double* d_scratch,
                 cudaStream_t stream = nullptr);

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV

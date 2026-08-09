/// @file cuda/planewave/internal/xc_pbe.cuh
/// @brief Real-space closed-shell PBE exchange-correlation on the plane-wave
///        grid — device kernel mirroring the per-point XC evaluation of
///        `cpu::pw::internal::XCBuilder::step(ws, XCType::PBE)`.
#pragma once

#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief Evaluate the closed-shell PBE XC potential (local part), energy
///        density, and GGA flux on the real grid.
///
/// Per grid point this reproduces `cpu::scf::evaluate_xc(XCType::PBE, ρ/2,
/// ρ/2, σ/4, σ/4, σ/4)` with σ = |∇ρ|² — PBE exchange (κ = 0.804,
/// μ = 0.21951…) over a PW92 LSDA correlation base plus the PBE H gradient
/// correction — including the `ρ < 1e-12` cutoff that zeroes every output.
/// Spin-unpolarized (ζ = 0 exactly, since ρ_α = ρ_β = ρ/2 bitwise): the spin
/// interpolation f(ζ), its derivative, ζ⁴ and dφ/dρ_σ all vanish identically
/// in IEEE arithmetic, so only the paramagnetic PW92 channel and the φ = 1
/// PBE correlation survive — the same exact collapse the ζ = 0 VWN5 kernel
/// documents in xc_lda.cuh, applied to the CPU's PBE branch term for term.
///
/// The kernel fuses the two per-point pieces of `XCBuilder::step`'s GGA grid
/// loop: the `evaluate_xc` call and the flux assembly
/// `flux_d = (vσ_aa + ½ vσ_ab) ∂_d ρ`. The gradient arrays are OVERWRITTEN
/// in place with the flux (each thread reads its own three gradient values
/// before writing them), because after this kernel the divergence pass needs
/// only the flux — mirroring how the CPU reuses its scratch lifetime, with
/// three fewer nr-sized device buffers.
///
/// The output `d_vxc_r` is the LOCAL part v_ρ only; the caller subtracts the
/// divergence of the flux (`v_xc = v_ρ − ∇·flux`) after its own FFT pass,
/// exactly as `XCBuilder::step` does.
///
/// @param d_rho_r  Real-space total density ρ(r), nr reals [electrons/Bohr³].
/// @param d_gx_r   In: ∂ρ/∂x [Bohr⁻⁴]; out: flux_x = vσ_eff·∂ρ/∂x [Hartree·Bohr⁻¹·Bohr⁻³].
/// @param d_gy_r   In: ∂ρ/∂y; out: flux_y. Same units as @p d_gx_r.
/// @param d_gz_r   In: ∂ρ/∂z; out: flux_z. Same units as @p d_gx_r.
/// @param d_vxc_r  Output local v_ρ(r) = ∂f/∂ρ_α, nr reals [Hartree].
/// @param d_exc_r  Output ε_xc(r) per electron, nr reals [Hartree].
/// @param nr       Number of real-space grid points.
/// @param stream   CUDA stream (default stream if null).
void xc_pbe_grid(const double* d_rho_r, double* d_gx_r, double* d_gy_r,
                 double* d_gz_r, double* d_vxc_r, double* d_exc_r, int nr,
                 cudaStream_t stream = nullptr);

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV

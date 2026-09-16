/// @file cpu/planewave/internal/xc_builder.h
/// @brief internal::XCBuilder — XC potential on the FFT grid via evaluate_xc.
///
/// Facade-private step helper owned by `cpu::pw::PW`. NOT a
/// `cpu::Component` subclass and never registered into a Plan.
///
/// Supports LDA (`XCType::LSDA`) and true GGA (`XCType::PBE` and other
/// `is_gga` functionals): spectral \(\nabla\rho\) from \(\rho(G)\), pointwise
/// libxc-parity `evaluate_xc`, and the closed-shell divergence term in
/// \(v_{\mathrm{xc}}\). Calling PBE with \(\sigma=0\) (LDA-limit) is **not**
/// the production GGA path.
///
/// Collinear spin (`step_spin`) uses the same true spin-GGA form as GPW
/// `xc_grid_gga_spin`: \(\sigma_{\sigma\sigma'}=(\nabla\rho_\sigma)\cdot(\nabla\rho_{\sigma'})\)
/// with spectral \(\nabla\rho_\sigma\), and
/// \(v_\sigma = v_{\rho_\sigma} - \nabla\cdot(2 v_{\sigma_{\sigma\sigma}}\nabla\rho_\sigma
/// + v_{\sigma_{\uparrow\downarrow}}\nabla\rho_{\bar\sigma})\).
#pragma once

#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/scf/xc_functional.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Computes V_xc(r), E_xc, and V_xc(G) from ρ(G).
///
/// FFTs ρ(G) → ρ(r), evaluates the XC functional pointwise (with spectral
/// density gradients for GGA), then FFTs V_xc(r) → V_xc(G). Scratch is
/// allocated in `prepare()`; `step()` performs no heap allocation beyond
/// workspace array resizes already owned by `PWWorkspace`.
class XCBuilder {
  public:
    /// @brief Size scratch to the workspace's FFT grid (idempotent).
    /// @param ws Plane-wave workspace (grid dims must be final).
    void prepare(const PWWorkspace& ws);

    /// @brief Compute `ws.V_xc_r`, `ws.exc_r`, `ws.E_xc`, `ws.V_xc_G`.
    /// @param ws Plane-wave workspace [ρ in Bohr⁻³, V/E in Hartree].
    /// @param xc Functional selector (`LSDA` = local; `PBE` = true GGA).
    void step(PWWorkspace& ws, cpu::scf::XCType xc = cpu::scf::XCType::PBE);

    /// @brief Collinear spin XC from `ws.rho_spin_G[0/1]`.
    ///        Writes `V_xc_spin_G`, `E_xc`, and `e_xc_int_spin`.
    ///        GGA functionals use true collinear spin-GGA (spectral ∇ρ_σ);
    ///        `XCType::LSDA` remains local-only.
    /// @param ws Plane-wave workspace [ρ_σ in Bohr⁻³, V/E in Hartree].
    /// @param xc Functional selector (default true GGA PBE).
    void step_spin(PWWorkspace& ws,
                   cpu::scf::XCType xc = cpu::scf::XCType::PBE);

  private:
    FFTWBuffer<fftw_complex> rho_g_half_;
    FFTWBuffer<AccReal> rho_r_;
    FFTWBuffer<fftw_complex> vxc_g_half_;
    // GGA scratch (sized in prepare; unused for pure LDA).
    // Restricted step + spin-up channel of step_spin reuse these three.
    FFTWBuffer<AccReal> grad_x_;
    FFTWBuffer<AccReal> grad_y_;
    FFTWBuffer<AccReal> grad_z_;
    FFTWBuffer<AccReal> flux_x_;
    FFTWBuffer<AccReal> flux_y_;
    FFTWBuffer<AccReal> flux_z_;
    FFTWBuffer<AccReal> div_r_;
    FFTWBuffer<fftw_complex> work_g_half_;
    // Collinear spin real-space densities / potentials.
    FFTWBuffer<AccReal> rho_up_r_;
    FFTWBuffer<AccReal> rho_dn_r_;
    FFTWBuffer<AccReal> vxc_up_r_;
    FFTWBuffer<AccReal> vxc_dn_r_;
    // Spin-GGA: down-channel gradients and both-channel fluxes for div.
    FFTWBuffer<AccReal> grad_dn_x_;
    FFTWBuffer<AccReal> grad_dn_y_;
    FFTWBuffer<AccReal> grad_dn_z_;
    FFTWBuffer<AccReal> flux_dn_x_;
    FFTWBuffer<AccReal> flux_dn_y_;
    FFTWBuffer<AccReal> flux_dn_z_;
    FFTWBuffer<AccReal> div_dn_r_;
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV

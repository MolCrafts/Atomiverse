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
#include "atomiverse/cpu/xc/xc_functional.h"
#include <array>
#include <span>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Computes V_xc(r), E_xc, and V_xc(G) from ρ(G).
///
/// FFTs ρ(G) → ρ(r), evaluates the XC functional pointwise (with spectral
/// density gradients for GGA), then FFTs V_xc(r) → V_xc(G).
/// Private quadrature uses 3/2 padding for LDA and 3x padding for GGA,
/// with the weighted adjoint Fourier projection back to the unchanged density
/// G-sphere. These fixed integration grids reduce nonlinear quadrature aliasing;
/// they are not a universal error bound for every density. Scratch is
/// allocated in `prepare()`; `step()` performs no heap allocation beyond
/// workspace array resizes already owned by `PWWorkspace`.
class XCBuilder {
  public:
    /// @brief Reserve fixed LDA/GGA XC quadrature plans and shared scratch (idempotent).
    /// @param ws Plane-wave workspace (grid dims must be final).
    void prepare(const PWWorkspace& ws);

    /// @brief Compute coarse `ws.V_xc_r`, `ws.E_xc`, and `ws.V_xc_G`.
    /// @param ws Plane-wave workspace [ρ in Bohr⁻³, V/E in Hartree].
    /// @param xc Functional selector (`LSDA` = local; `PBE` = true GGA).
    void step(PWWorkspace& ws, cpu::xc::XCType xc = cpu::xc::XCType::PBE);

    /// @brief Collinear spin XC from `ws.rho_spin_G[0/1]`.
    ///        Writes `V_xc_spin_G`, `E_xc`, and `e_xc_int_spin`.
    ///        GGA functionals use true collinear spin-GGA (spectral ∇ρ_σ);
    ///        `XCType::LSDA` remains local-only.
    /// @param ws Plane-wave workspace [ρ_σ in Bohr⁻³, V/E in Hartree].
    /// @param xc Functional selector (default true GGA PBE).
    void step_spin(PWWorkspace& ws, cpu::xc::XCType xc = cpu::xc::XCType::PBE);

    /// Physical Fourier inputs and preallocated coarse-grid response output.
    struct ResponseChannel {
        std::span<const std::complex<AccReal>> reference;
        std::span<const std::complex<AccReal>> variation;
        std::span<AccReal> potential;
    };
    /// Central difference of the same fixed-grid potential, with caller's
    /// unchanged epsilon. One channel denotes total restricted density; two
    /// denote independent alpha/beta densities. Does not mutate accepted ws.
    void response(const PWWorkspace& ws, cpu::xc::XCType xc,
                  std::span<const ResponseChannel> channels, AccReal epsilon);

    /// Extensive XC virial W=-dE_xc/dstrain [Ha], xx,yy,zz,xy,xz,yz,
    /// from the last completed ordinary step/step_spin at that evaluation's
    /// geometry and exact padded quadrature. Not the total PW stress.
    /// prepare/response and incomplete evaluations preserve this value.
    const std::array<AccReal, 6>& virial_ha() const { return virial_ha_; }

  private:
    std::array<AccReal, 6> virial_ha_{};
    /// Shape-only plan; physical G vectors and cell volume are read each call.
    std::unique_ptr<FFTPlan3D> local_quadrature_fft_;
    std::unique_ptr<FFTPlan3D> gradient_quadrature_fft_;
    FFTWBuffer<AccReal> restricted_vxc_r_;
    void check_prepared(const PWWorkspace& ws) const;
    void project_to_base(const PWWorkspace& ws,
                         const xt::xarray<std::complex<AccReal>>& potential,
                         std::span<AccReal> output);
    void cache_response_reference(const PWWorkspace& ws, cpu::xc::XCType xc,
                                  std::span<const ResponseChannel> channels);
    // Fixed reference fields are interpolated once. Keys contain physical
    // coefficients/modes/metric and quadrature shape, never workspace pointers.
    int response_channels_ = 0;
    std::array<int, 3> response_shape_{};
    std::array<std::vector<std::complex<AccReal>>, 2> response_reference_;
    std::vector<int> response_h_, response_k_, response_l_;
    std::vector<Vec3> response_metric_;
    std::array<FFTWBuffer<AccReal>, 2> response_density_;
    std::array<std::array<FFTWBuffer<AccReal>, 3>, 2> response_gradient_;
    std::array<xt::xarray<std::complex<AccReal>>, 2> response_potential_G_;
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

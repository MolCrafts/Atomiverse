/// @file cpu/pw/internal/xc_builder.h
/// @brief internal::XCBuilder — XC potential on the FFT grid via evaluate_xc.
///
/// Facade-private step helper owned by `cpu::pw::PWSolver`. NOT a
/// `cpu::Component` subclass and never registered into a Plan.
#pragma once

#include "atomiverse/cpu/pw/internal/pw_workspace.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Computes V_xc(r), E_xc, and V_xc(G) from ρ(G).
///
/// FFTs ρ(G) → ρ(r), evaluates the XC functional pointwise, then FFTs
/// V_xc(r) → V_xc(G). All scratch buffers are allocated in `prepare()`;
/// `step()` performs no allocation.
class XCBuilder {
  public:
    /// @brief Size scratch to the workspace's FFT grid (idempotent).
    /// @param ws Plane-wave workspace (grid dims must be final).
    void prepare(const PWWorkspace& ws);

    /// @brief Compute `ws.V_xc_r`, `ws.exc_r`, `ws.E_xc`, `ws.V_xc_G`.
    /// @param ws Plane-wave workspace [ρ in Bohr⁻³, V/E in Hartree].
    void step(PWWorkspace& ws);

  private:
    FFTWBuffer<fftw_complex> rho_g_half_;
    FFTWBuffer<AccReal> rho_r_;
    FFTWBuffer<fftw_complex> vxc_g_half_;
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV

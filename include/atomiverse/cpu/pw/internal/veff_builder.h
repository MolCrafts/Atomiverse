/// @file cpu/pw/internal/veff_builder.h
/// @brief internal::VeffBuilder — assemble V_eff = V_loc + V_H + V_xc.
///
/// Facade-private step helper owned by `cpu::pw::PWSolver`. NOT a
/// `cpu::Component` subclass and never registered into a Plan.
#pragma once

#include "atomiverse/cpu/pw/internal/pw_workspace.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Assembles V_eff(G) and FFTs it to the real-space grid V_eff(r)
///        for the FFT-based H·ψ application in the Davidson solver.
class VeffBuilder {
  public:
    /// @brief Size scratch to the workspace's FFT grid (idempotent).
    void prepare(const PWWorkspace& ws);

    /// @brief Compute `ws.V_eff_G` and `ws.V_eff_r` [Hartree].
    void step(PWWorkspace& ws);

  private:
    FFTWBuffer<fftw_complex> vg_;
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV

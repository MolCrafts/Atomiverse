/// @file cpu/vibration/phonon_diagonalizer.h
/// @brief Crystal Γ-point diagonalizer: full \(3N\) mass-weighted spectrum.
///
/// Unlike `HessianDiagonalizer` (molecular TR projection → \(3N-6/5\)), this
/// keeps all \(3N\) modes. At an equilibrium crystal the three acoustic
/// branches sit near zero; optical branches are physical phonons.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/vibration/hessian_workspace.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Structural component: full \(3N\) dynamical-matrix eigensolve.
class PhononDiagonalizer : public cpu::Component {
  public:
    /// @param ws Non-owning pointer to the shared Hessian workspace.
    explicit PhononDiagonalizer(HessianWorkspace* ws) : ws_(ws) {}

    std::string name() const override { return "phonon_diagonalizer"; }

    /// @brief Default binding: `vib::Phase::diagonalize` @ +1 (after MassWeighting).
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::vib::Phase::diagonalize, +1)};
    }

    /// @brief Diagonalize `hessian_mw` on the finalize pass; write frequencies.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    HessianWorkspace* ws_;
};

} // namespace vib
} // namespace cpu
} // namespace ATV

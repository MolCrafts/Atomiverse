/// @file cpu/observe/classical_polarization.h
/// @brief Classical polarization density from Frame charges and positions.
///
/// MetalWalls / Dufils finite-field observable:
/// \[
///   \mathbf{P} = \frac{1}{V}\sum_i q_i\,\mathbf{r}_i^{\mathrm{u}}
///   \quad [e/\mathrm{Å}^2]
/// \]
/// with unwrapped orthorhombic coordinates. Frame-immutable Observer —
/// does not mutate charges or positions. No public Config bag.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Classical polarization density \(\mathbf{P}=V^{-1}\sum q\mathbf{r}^{u}\).
///
/// Default binding: `md::Phase::end_of_step` after charge solve / forces.
/// Read `Px()`, `Py()`, `Pz()` or `P()` after launch [e/Å²].
class ClassicalPolarization : public Observer {
  public:
    ClassicalPolarization() = default;

    /// @brief Component identifier.
    /// @return `"classical_polarization"`.
    std::string name() const override { return "classical_polarization"; }

    /// @brief Default phase: end of MD step.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +50)};
    }

    /// @brief Compute \(\mathbf{P}\) from owned atoms (orthorhombic unwrap).
    /// @param ctx Launch context (frame + optional domain decomposition).
    void launch(const LaunchContext& ctx) override;

    /// @brief \(P_x\) from last launch [e/Å²].
    double Px() const noexcept { return P_[0]; }
    /// @brief \(P_y\) from last launch [e/Å²].
    double Py() const noexcept { return P_[1]; }
    /// @brief \(P_z\) from last launch [e/Å²].
    double Pz() const noexcept { return P_[2]; }
    /// @brief Full polarization vector from last launch [e/Å²].
    std::array<double, 3> P() const noexcept { return P_; }

    /// @brief Total dipole \(\boldsymbol{\mu}=\sum q\mathbf{r}^{u}\) [e·Å]
    ///        before dividing by volume.
    std::array<double, 3> dipole() const noexcept { return mu_; }

  private:
    std::array<double, 3> P_{{0.0, 0.0, 0.0}};
    std::array<double, 3> mu_{{0.0, 0.0, 0.0}};
};

} // namespace md
} // namespace cpu
} // namespace ATV

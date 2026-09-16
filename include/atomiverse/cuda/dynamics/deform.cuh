/// @file cuda/dynamics/deform.cuh
/// @brief Deform -- uniaxial strain deformation component.
///
/// Rescales box and atom positions along a given axis at a constant engineering strain rate.
/// CUDA-free header.
#pragma once

#include <string>

#include "atomiverse/core/role.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Uniaxial constant-strain-rate deformation.
///
/// Each step: box[axis] *= (1 + rate*dt). Atom positions remapped proportionally.
/// Binds to initial_integrate at order -100 (before VelocityVerlet).
class Deform : public Component {
  public:
    /// @brief Construct deformation component.
    /// @param axis        Deformation axis (0=x, 1=y, 2=z).
    /// @param strain_rate Engineering strain rate [1/fs].
    /// @param dt          Time step [fs] (must match integrator).
    Deform(int axis, double strain_rate, double dt);

    std::string name() const override { return "deform"; }
    Role role() const override { return 0; }

    /// @brief Structural role — a box/position time-step integrator.
    ///
    /// Mirrors `cpu::md::Deform`: a strain-rate deformation advances the
    /// box and positions each step, so it occupies the `Integrator`
    /// slot.
    /// @return ComponentRole::Integrator.
    /* role via Integrator base */
    /// @brief Write-set: affine remap of coordinates as the box strains.

    std::vector<DefaultBinding> default_bindings() const override;
    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

  private:
    int axis_;
    double strain_rate_, dt_;
    // LAMMPS-equivalent engineering-strain bookkeeping: cache L0 at build()
    // and compute L_k = L0·(1 + erate·k·dt) per step, so the box grows
    // LINEARLY with time. The previous per-step multiplier `(1+erate·dt)`
    // compounded to L0·exp(erate·t) — fine for small total strain (LAMMPS
    // agreement at ε≪1) but wrong at the stage-5 full strain ε=1.7 of the
    // MSU PE deck, where it overshot to L/L0 ≈ 5.57 vs LAMMPS 2.72.
    double L0_ = 0.0;
    long long step_count_ = 0;
};

} // namespace md
} // namespace cuda
} // namespace ATV

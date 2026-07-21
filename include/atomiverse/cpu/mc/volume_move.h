/// @file cpu/mc/volume_move.h
/// @brief cpu::mc::VolumeMove — isotropic volume-change Mover for NPT-MC.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/mover.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace mc {

/// @brief Monte Carlo trial move: change the volume by a uniform draw
///        `ΔV ∈ [-δV, δV]` [Å³] and rescale the box + every coordinate
///        isotropically (Frenkel & Smit §5.4.2).
///
/// Sampling **uniformly in V** pairs with the `NptMetropolis` rule's
/// `N k_B T ln(V'/V)` term (a ln-V walk would need the (N+1)-form instead).
/// Records the original box AND every atom row into the Frame patch, so a
/// rejection restores coordinates bitwise (`x*s/s != x` in floating point).
///
/// A draw that would make the volume non-positive stamps an empty
/// (`Verb::None`) proposal, which the policy rejects without counting a
/// Boltzmann draw.
///
/// Role `Mover`, capabilities `None`. Bound to `sampling::Phase::propose`.
class VolumeMove : public cpu::Mover {
  public:
    /// @param dv_max Volume half-width δV [Å³] (must be > 0).
    /// @throws std::invalid_argument if `dv_max <= 0`.
    explicit VolumeMove(double dv_max);

    /// @brief Component identifier.
    /// @return "volume".
    std::string name() const override { return "volume"; }

    /// @brief Structural role.
    /// @return `ComponentRole::Mover`.
    /* role via Mover base */

    /// @brief Default binding: the `propose` phase, order 0.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::sampling::Phase::propose, 0)};
    }

    /// @brief Propose one volume change (or nothing on the bootstrap pass).
    /// @param ctx Launch context; reads `runtime.mc.rng`, writes the frame
    ///        and `runtime.mc.proposal`.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    double dv_max_; ///< Volume half-width δV [Å³].
};

} // namespace mc
} // namespace cpu
} // namespace ATV

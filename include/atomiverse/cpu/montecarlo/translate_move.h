/// @file cpu/mc/translate_move.h
/// @brief cpu::mc::TranslateMove — single-particle symmetric displacement Mover.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/mover.h"

namespace ATV {
namespace cpu {
namespace mc {

/// @brief Monte Carlo trial move: displace one random atom by a uniform
///        vector in the symmetric box `[-δ, δ]³` [Å].
///
/// The symmetric (uniform) proposal makes forward and reverse proposal
/// densities equal, so acceptance reduces to the energy-only Metropolis rule
/// the sampling driver applies (detailed balance for the canonical ensemble;
/// Frenkel & Smit ch. 3). Bound to `sampling::Phase::propose`.
///
/// On `launch()` it draws the atom index and the per-axis displacement from
/// the driver-owned RNG published on `Context::rng` (so the whole trajectory
/// is reproducible from one seed), mutates the frame in place, and records the
/// changeset into `Context::proposal` for the driver to commit or roll back.
/// On the driver's energy-only bootstrap pass (`step < 0`) it proposes nothing
/// so the evaluator sees the unperturbed initial configuration. On an empty
/// cell (`nlocal() == 0`, reachable under GCMC once the chain has deleted its
/// way down to N = 0) there is no atom to pick, so it proposes nothing either
/// and stamps `Transaction::Verb::None` for the policy to reject.
///
/// Role `Mover`, capabilities `None` (it moves atoms; it writes no energy /
/// force field).
class TranslateMove : public cpu::Mover {
  public:
    /// @param delta Displacement half-width δ [Å] (must be > 0). Each axis is
    ///        displaced by a uniform draw in `[-δ, δ]`.
    /// @throws std::invalid_argument if `delta <= 0`.
    explicit TranslateMove(double delta);

    /// @brief Component identifier.
    /// @return "translate".
    std::string name() const override { return "translate"; }

    /// @brief Structural role.
    /// @return `ComponentRole::Mover`.
    /* role via Mover base */

    /// @brief Default binding: the `propose` phase, order 0.
    /// @return One binding on `sampling::Phase::propose`.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::sampling::Phase::propose, 0)};
    }

    /// @brief Propose one displacement (or nothing on the bootstrap pass).
    /// @param ctx Launch context; reads `runtime.mc.rng`, writes the frame and
    ///        `runtime.mc.proposal`.
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    double delta_; ///< Displacement half-width δ [Å].
};

} // namespace mc
} // namespace cpu
} // namespace ATV

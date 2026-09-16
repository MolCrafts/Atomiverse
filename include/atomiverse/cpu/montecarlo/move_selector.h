/// @file cpu/montecarlo/move_selector.h
/// @brief cpu::mc::MoveSelector — draws ONE trial-move type per trial.
#pragma once

#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/mover.h"

namespace ATV {
namespace cpu {
namespace mc {

/// @brief A Mover of Movers: each trial it draws which child move type to
///        attempt, with the configured weights, and fires exactly that one.
///
/// Registering two Movers side by side on the `propose` phase would fire BOTH
/// every trial — one trial per iteration is the Monte Carlo contract, so a
/// mixed-move ensemble (NPT: displacements + volume moves; μVT: displacements
/// + insert/delete) goes through this selector instead:
///
/// @code{.cpp}
///   mc.set_mover(cpu::mc::MoveSelector{}
///                    .add(cpu::mc::TranslateMove{0.2}, 0.9)
///                    .add(cpu::mc::VolumeMove{50.0}, 0.1));
/// @endcode
///
/// Weights are relative (normalized internally). The type-selection draw
/// comes from the run's single RNG stream (`ctx.mc.rng`), interleaved with
/// the child's own draws — one seed, one trajectory. Per-verb
/// attempt/acceptance tallies are reported by the driver
/// (`McBase::attempts(verb)` / `accepts(verb)`), which sees each trial's verb
/// on the proposal.
///
/// Children are held by `shared_ptr` so the selector stays copyable through
/// the driver's forwarding setters; a selector is configured once and never
/// mutated after registration.
class MoveSelector : public cpu::Mover {
  public:
    /// @brief Construct an empty selector; populate with `add`.
    MoveSelector() = default;

    /// @brief Register a child move type with a relative weight.
    ///
    /// Ref-qualified: chaining on a temporary (`MoveSelector{}.add(…).add(…)`)
    /// stays an rvalue all the way into `set_mover`, so the driver takes the
    /// OWNING branch of `PhaseBuilder::add`. Without the `&&` overload the chain
    /// yields an lvalue reference to the temporary, the driver stores a
    /// non-owning pointer to it, and the selector dangles the moment the
    /// full expression ends.
    ///
    /// @param m Mover component (moved in).
    /// @param weight Relative draw weight (must be > 0).
    /// @return `*this`, for chaining.
    /// @throws std::invalid_argument on a non-positive weight or a non-Mover.
    template <typename M>
    MoveSelector& add(M&& m, double weight) & {
        add_(std::forward<M>(m), weight);
        return *this;
    }

    /// @brief Rvalue-chain overload of `add` (see above).
    template <typename M>
    MoveSelector&& add(M&& m, double weight) && {
        add_(std::forward<M>(m), weight);
        return std::move(*this);
    }

    /// @brief Component identifier.
    /// @return "move_selector".
    std::string name() const override { return "move_selector"; }

    /// @brief Structural role.
    /// @return `ComponentRole::Mover`.
    /* role via Mover base */

    /// @brief Default binding: the `propose` phase, order 0.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::sampling::Phase::propose, 0)};
    }

    /// @brief Forward one-time allocation to every child.
    /// @throws std::runtime_error if no child was added.
    void build(const cpu::BuildContext& ctx) override {
        if (children_.empty())
            throw std::runtime_error(
                "MoveSelector: no child movers — add(mover, weight) at least "
                "one");
        for (auto& [w, child] : children_)
            child->build(ctx);
    }

    /// @brief Draw one child by weight and fire it (nothing on the bootstrap
    ///        pass — no draw, so the RNG stream stays aligned with a
    ///        single-mover run).
    /// @param ctx Launch context; reads `runtime.mc.rng`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (ctx.step < 0)
            return;
        std::mt19937_64* rng = ctx.runtime.mc.rng;
        if (rng == nullptr) {
            throw std::runtime_error(
                "MoveSelector::launch: ctx.mc.rng not published — "
                "MoveSelector must run under an MC preset");
        }
        std::uniform_real_distribution<double> u(0.0, total_weight_);
        const double draw = u(*rng);
        for (auto& [cumulative, child] : children_) {
            if (draw < cumulative) {
                child->launch(ctx);
                return;
            }
        }
        children_.back().second->launch(ctx); // draw == total (edge)
    }

    /// @brief Forward teardown to every child.
    void destroy(const cpu::DestroyContext& ctx) override {
        for (auto& [w, child] : children_)
            child->destroy(ctx);
    }

  private:
    /// @brief Shared body of the two ref-qualified `add` overloads.
    template <typename M>
    void add_(M&& m, double weight) {
        if (!(weight > 0.0))
            throw std::invalid_argument("MoveSelector: weight must be > 0");
        auto child = std::make_shared<std::decay_t<M>>(std::forward<M>(m));
        if (dynamic_cast<const Mover*>(child.get()) == nullptr)
            throw std::invalid_argument(
                "MoveSelector: child must be a cpu::Mover");
        total_weight_ += weight;
        children_.emplace_back(total_weight_, std::move(child));
    }

    /// Children with their CUMULATIVE weight (draw < cumulative selects).
    std::vector<std::pair<double, std::shared_ptr<cpu::Component>>> children_;
    double total_weight_ = 0.0; ///< Sum of all registered weights.
};

} // namespace mc
} // namespace cpu
} // namespace ATV

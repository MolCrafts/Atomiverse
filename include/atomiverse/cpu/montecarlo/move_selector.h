/// @file cpu/montecarlo/move_selector.h
/// @brief cpu::mc::MoveSelector — draws ONE trial-move type per trial.
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
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
/// (`MC::attempts(verb)` / `accepts(verb)`), which sees each trial's verb
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
        requires std::derived_from<std::decay_t<M>, Mover>
    MoveSelector& add(M&& m, double weight) & {
        add_child(std::make_shared<std::decay_t<M>>(std::forward<M>(m)),
                  weight);
        return *this;
    }

    /// @brief Rvalue-chain overload of `add` (see above).
    template <typename M>
        requires std::derived_from<std::decay_t<M>, Mover>
    MoveSelector&& add(M&& m, double weight) && {
        add_child(std::make_shared<std::decay_t<M>>(std::forward<M>(m)),
                  weight);
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
        if (active_)
            throw std::logic_error("MoveSelector: already active in a run");
        std::unordered_set<const Mover*> seen;
        validate_tree_(seen);
        std::size_t built = 0;
        try {
            for (const auto& entry : children_) {
                entry.second->build(ctx);
                ++built;
            }
        } catch (...) {
            while (built > 0) {
                try {
                    children_[--built].second->destroy(
                        {ctx.runtime, ctx.frame});
                } catch (...) {
                    // Keep the original build error.
                }
            }
            throw;
        }
        active_ = true;
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
        std::exception_ptr failure;
        for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
            try {
                it->second->destroy(ctx);
            } catch (...) {
                if (!failure)
                    failure = std::current_exception();
            }
        }
        active_ = false;
        if (failure)
            std::rethrow_exception(failure);
    }

    bool changes_connectivity() const override {
        return std::ranges::any_of(children_, [](const auto& entry) {
            return entry.second->changes_connectivity();
        });
    }

    bool requires_full_frame() const override {
        return std::ranges::any_of(children_, [](const auto& entry) {
            return entry.second->requires_full_frame();
        });
    }

    bool requires_tag_index() const override {
        return std::ranges::any_of(children_, [](const auto& entry) {
            return entry.second->requires_tag_index();
        });
    }

    /// Register a polymorphic mover with shared ownership.
    MoveSelector& add_child(std::shared_ptr<Mover> child, double weight) {
        if (active_)
            throw std::logic_error(
                "MoveSelector: cannot add movers during a run");
        if (!child || contains_(child.get())
            || (dynamic_cast<const MoveSelector*>(child.get()) != nullptr
                && static_cast<const MoveSelector&>(*child).contains_(this)))
            throw std::invalid_argument(
                "MoveSelector: a mover must be non-null and may not be "
                "repeated or create a cycle");
        const double total = total_weight_ + weight;
        if (!std::isfinite(total) || !(total > total_weight_))
            throw std::invalid_argument(
                "MoveSelector: weight must be finite, positive and "
                "representable in the sum");
        children_.emplace_back(total, std::move(child));
        total_weight_ = total;
        return *this;
    }

  private:
    void validate_tree_(std::unordered_set<const Mover*>& seen) const {
        if (!seen.insert(this).second)
            throw std::invalid_argument(
                "MoveSelector: repeated selector instance");
        for (const auto& entry : children_) {
            const auto* child = entry.second.get();
            if (const auto* selector = dynamic_cast<const MoveSelector*>(child))
                selector->validate_tree_(seen);
            else if (!seen.insert(child).second)
                throw std::invalid_argument(
                    "MoveSelector: repeated mover instance");
        }
    }

    bool contains_(const Mover* needle) const {
        if (needle == this)
            return true;
        for (const auto& entry : children_) {
            const auto* child = entry.second.get();
            if (child == needle)
                return true;
            const auto* selector = dynamic_cast<const MoveSelector*>(child);
            if (selector && selector->contains_(needle))
                return true;
        }
        return false;
    }

    /// Children with their cumulative weight (draw < cumulative selects).
    std::vector<std::pair<double, std::shared_ptr<Mover>>> children_;
    bool active_ = false;
    double total_weight_ = 0.0; ///< Sum of all registered weights.
};

} // namespace mc
} // namespace cpu
} // namespace ATV

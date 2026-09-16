/// @file cpu/driver/nested_loop_adapter.h
/// @brief Host adapter: a DriverNode contributes as one Component.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/driver/driver_node.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/column.h"

namespace ATV {
namespace cpu {

/// @brief Wraps a nested loop so the outer Plan only sees a Component.
///
/// Inner phases stay inside `run_nested_erased`. This type is a Component,
/// never a DriverNode.
class NestedLoopAdapter : public Component {
  public:
    /// @brief Borrow `inner` for the adapter lifetime.
    /// @param inner Nested loop node (caller retains ownership).
    /// @param budget Iteration budget; `< 0` uses `inner.default_budget()`.
    NestedLoopAdapter(DriverNode& inner, int budget = -1)
        : inner_(&inner), budget_(resolve_budget(inner, budget)) {}

    /// @brief Take ownership of `inner`.
    /// @param inner Non-null nested loop node.
    /// @param budget Iteration budget; `< 0` uses `inner.default_budget()`.
    /// @throws std::invalid_argument if `inner` is null.
    NestedLoopAdapter(std::unique_ptr<DriverNode> inner, int budget = -1)
        : owned_(std::move(inner)), inner_(owned_.get()) {
        if (!inner_) {
            throw std::invalid_argument(
                "NestedLoopAdapter: null DriverNode. Pass a node or use "
                "the borrow constructor.");
        }
        budget_ = resolve_budget(*inner_, budget);
    }

    /// @brief Own a concrete DriverNode rvalue that is not a Component.
    template <
        typename Node,
        std::enable_if_t<
            std::is_base_of_v<DriverNode, std::remove_reference_t<Node>>
                && !std::is_base_of_v<Component, std::remove_reference_t<Node>>
                && !std::is_lvalue_reference_v<Node>,
            int> = 0>
    NestedLoopAdapter(Node&& node, int budget = -1)
        : NestedLoopAdapter(std::unique_ptr<DriverNode>(
                                std::make_unique<std::remove_reference_t<Node>>(
                                    std::forward<Node>(node))),
                            budget) {}

    NestedLoopAdapter(const NestedLoopAdapter&) = delete;
    NestedLoopAdapter& operator=(const NestedLoopAdapter&) = delete;
    NestedLoopAdapter(NestedLoopAdapter&&) noexcept = default;
    NestedLoopAdapter& operator=(NestedLoopAdapter&&) noexcept = default;

    std::string name() const override { return "nested_loop_adapter"; }

    bool provides_energy() const override {
        if (const auto* hf = dynamic_cast<const HF*>(inner_))
            return hf->provides_energy();
        return false;
    }
    bool provides_force() const override {
        if (const auto* hf = dynamic_cast<const HF*>(inner_))
            return hf->provides_force();
        return false;
    }

    /// @brief The wrapped node (for host walks that must see HF).
    DriverNode* inner() noexcept { return inner_; }
    const DriverNode* inner() const noexcept { return inner_; }

    bool requires_tag_index() const override {
        return inner_ && inner_->requires_tag_index();
    }

    void build(const BuildContext& ctx) override {
        inner_->build_on(require_runtime(ctx.runtime, "build"));
    }
    void launch(const LaunchContext& ctx) override {
        const auto saved_window = ctx.frame.resolution_window();
        const bool saved_forces = ctx.runtime.exec.forces_requested;
        ctx.frame.restore_resolution_window(ResolutionWindow{});
        ctx.runtime.exec.forces_requested =
            saved_forces && ctx.forces_requested;
        inner_->run_nested_erased(ctx.runtime, ctx.frame, budget_);
        ctx.runtime.exec.forces_requested = saved_forces;
        ctx.frame.restore_resolution_window(saved_window);
    }
    void destroy(const DestroyContext& ctx) override {
        inner_->destroy_on(require_runtime(ctx.runtime, "destroy"));
    }

  private:
    static int resolve_budget(DriverNode& inner, int budget) {
        return budget < 0 ? inner.default_budget() : budget;
    }

    static Runtime& require_runtime(Context& ctx, const char* where) {
        if (!ctx.exec.runtime) {
            throw std::runtime_error(
                std::string("NestedLoopAdapter::") + where
                + ": no Runtime on the Context. Bind the adapter under a "
                  "parent Driver that has completed setup_pass.");
        }
        return *ctx.exec.runtime;
    }

    std::unique_ptr<DriverNode> owned_;
    DriverNode* inner_ = nullptr;
    int budget_ = 0;
};

} // namespace cpu
} // namespace ATV

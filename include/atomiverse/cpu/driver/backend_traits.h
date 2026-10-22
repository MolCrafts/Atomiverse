/// @file cpu/driver/backend_traits.h
/// @brief CPU backend_traits specialization. CUDA-free header.
#pragma once

#include "atomiverse/core/plan.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/cpu/component/timestep_protocol.h"
#include "atomiverse/cpu/driver/driver_node.h"
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "atomiverse/core/backend_traits.h"
#include "atomiverse/core/device.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/driver/runtime.h"

namespace ATV {

/// @brief CPU backend trait specialization.
template <>
struct backend_traits<cpu::device_tag> {
    using component_type = cpu::Component;
    using backend_type = cpu::Runtime;
    using frame_type = cpu::Frame;
    using driver_node_type = cpu::DriverNode;
    using timer_type = TimerRegistry;
    static constexpr bool has_device_timing = false;

    static void run(backend_type& backend, frame_type& frame,
                    const Plan<component_type>& plan, timer_type*, int steps) {
        backend.init();
        backend.load(plan);
        backend.set_frame(frame);
        backend.build_components();
        backend.run(steps);
        backend.destroy_components();
        backend.shutdown();
    }

    template <typename Node>
    static std::pair<bool, bool> node_outputs(const Node& node) {
        // HF can be a sibling of DriverNode. A virtual query through the
        // tag preserves its defaults and concrete overrides without ambiguity.
        if constexpr (std::is_base_of_v<cpu::HF, Node>) {
            return {static_cast<const cpu::HF&>(node).provides_energy(),
                    static_cast<const cpu::HF&>(node).provides_force()};
        } else {
            bool energy = false, force = false;
            if constexpr (requires { node.provides_energy(); })
                energy = node.provides_energy();
            if constexpr (requires { node.provides_force(); })
                force = node.provides_force();
            return {energy, force};
        }
    }

    template <typename Node>
    static std::unique_ptr<component_type>
    nested_slot(std::unique_ptr<Node> owned, Node* node, int cap, bool, bool) {
        cpu::TimestepConsumer* step = nullptr;
        if constexpr (std::is_base_of_v<cpu::TimestepConsumer, Node>)
            step = static_cast<cpu::TimestepConsumer*>(node);
        else if constexpr (requires { node->timestep_consumer(); })
            step = node->timestep_consumer();

        struct Slot : component_type {
            std::unique_ptr<cpu::DriverNode> owned_node;
            Node* node = nullptr;
            cpu::TimestepConsumer* step = nullptr;
            int budget = 0;

            std::string name() const override { return "nested_driver"; }
            cpu::DriverNode* nested_node() const override { return node; }
            bool provides_energy() const override {
                return backend_traits::node_outputs(*node).first;
            }
            bool provides_force() const override {
                return backend_traits::node_outputs(*node).second;
            }
            bool requires_full_frame() const override {
                return node->requires_full_frame();
            }
            bool requires_tag_index() const override {
                return node && node->requires_tag_index();
            }
            cpu::TimestepConsumer* as_timestep_consumer() override {
                return step;
            }
            void build(const cpu::BuildContext& ctx) override {
                node->build_on(require_runtime(ctx.runtime, "build"));
            }
            void launch(const cpu::LaunchContext& ctx) override {
                const auto saved_window = ctx.frame.resolution_window();
                const bool saved_forces = ctx.runtime.exec.forces_requested;
                const bool saved_stress = ctx.runtime.exec.stress_requested;
                struct Restore {
                    const cpu::LaunchContext& ctx;
                    cpu::ResolutionWindow window;
                    bool forces;
                    bool stress;
                    ~Restore() {
                        ctx.runtime.exec.forces_requested = forces;
                        ctx.runtime.exec.stress_requested = stress;
                        ctx.frame.restore_resolution_window(window);
                    }
                } restore{ctx, saved_window, saved_forces, saved_stress};
                ctx.frame.restore_resolution_window(cpu::ResolutionWindow{});
                ctx.runtime.exec.forces_requested =
                    saved_forces && ctx.forces_requested;
                ctx.runtime.exec.stress_requested =
                    saved_stress && ctx.stress_requested;
                node->run_nested_erased(ctx.runtime, ctx.frame, budget);
            }
            void destroy(const cpu::DestroyContext& ctx) override {
                node->destroy_on(require_runtime(ctx.runtime, "destroy"));
            }

            static cpu::Runtime& require_runtime(cpu::Context& ctx,
                                                 const char* where) {
                if (!ctx.exec.runtime) {
                    throw std::runtime_error(
                        std::string("nested driver ") + where
                        + ": no Runtime on the Context. Bind the driver under "
                          "a parent Driver that has completed setup_pass.");
                }
                return *ctx.exec.runtime;
            }
        };

        auto slot = std::make_unique<Slot>();
        slot->owned_node = std::move(owned);
        slot->node = node;
        slot->step = step;
        slot->budget = cap;
        return slot;
    }
};

} // namespace ATV

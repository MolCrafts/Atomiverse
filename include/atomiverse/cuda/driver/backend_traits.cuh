/// @file cuda/driver/backend_traits.cuh
/// @brief CUDA backend_traits specialization.
#pragma once

#include "atomiverse/core/plan.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/device/timer.h"
#include "atomiverse/cuda/driver/driver_node.h"
#include <memory>
#include <string>
#include <utility>

#include "atomiverse/core/backend_traits.h"
#include "atomiverse/core/device.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/driver/backend.cuh"

namespace ATV {

/// @brief CUDA backend trait specialization.
template <>
struct backend_traits<cuda::device_tag> {
    using component_type = cuda::Component;
    using backend_type = cuda::Backend;
    using frame_type = cpu::Frame;
    using driver_node_type = cuda::DriverNode;
    using timer_type = cuda::Timer;
    static constexpr bool has_device_timing = true;

    static void run(backend_type& backend, frame_type& frame,
                    const Plan<component_type>& plan, timer_type* timer,
                    int steps) {
        backend.init();
        if (timer)
            backend.set_timer(timer);
        backend.load(plan);
        backend.materialize(frame);
        backend.build_components();
        backend.run(steps);
        backend.destroy_components();
        backend.retrieve(frame);
        backend.shutdown();
    }

    template <typename Node>
    static std::pair<bool, bool> node_outputs(const Node& node) {
        bool energy = false, force = false;
        if constexpr (requires { node.provides_energy(); })
            energy = node.provides_energy();
        if constexpr (requires { node.provides_force(); })
            force = node.provides_force();
        return {energy, force};
    }

    template <typename Node>
    static std::unique_ptr<component_type>
    nested_slot(std::unique_ptr<Node> owned, Node* node, int cap, bool, bool) {
        struct Slot : component_type {
            std::unique_ptr<cuda::DriverNode> owned_node;
            Node* node = nullptr;
            int budget = 0;

            std::string name() const override { return "nested_driver"; }
            bool provides_energy() const override {
                return backend_traits::node_outputs(*node).first;
            }
            bool provides_force() const override {
                return backend_traits::node_outputs(*node).second;
            }
            bool provides_system_stress() const override {
                if constexpr (requires { node->provides_system_stress(); })
                    return node->provides_system_stress();
                return false;
            }
            bool requires_system_stress() const override {
                if constexpr (requires { node->requires_system_stress(); })
                    return node->requires_system_stress();
                return false;
            }
            bool supports_input_changes(InputChange changes) const override {
                return node->supports_input_changes(changes);
            }
            void invalidate_inputs(InputChange changes,
                                   const cuda::BuildContext& ctx) override {
                node->invalidate_inputs(changes, ctx.runtime, ctx.frame);
            }
            void build(const cuda::BuildContext& ctx) override {
                node->build_on(ctx.runtime, ctx.frame);
            }
            void launch(const cuda::LaunchContext& ctx) override {
                struct Restore {
                    cuda::Context& runtime;
                    cuda::ExecServices exec;
                    cuda::LoopServices loop;
                    ~Restore() {
                        runtime.exec = exec;
                        runtime.loop = loop;
                    }
                } restore{ctx.runtime, ctx.runtime.exec, ctx.runtime.loop};
                auto& exec = ctx.runtime.exec;
                exec.forces_requested &= ctx.forces_requested;
                exec.stress_requested &=
                    ctx.stress_requested && provides_system_stress();
                exec.initializing_forces |= ctx.initializing_forces;
                exec.trial_evaluation |= ctx.trial_evaluation;
                node->run_nested_erased(ctx.runtime, ctx.frame, budget);
            }
            void destroy(const cuda::DestroyContext& ctx) override {
                node->destroy_on(ctx.runtime, ctx.frame);
            }
        };

        auto slot = std::make_unique<Slot>();
        slot->owned_node = std::move(owned);
        slot->node = node;
        slot->budget = cap;
        return slot;
    }
};

} // namespace ATV

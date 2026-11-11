/// @file cpu/electronic/aimd_method.h
/// @brief CarParrinelloMethod / XlBomdMethod — AIMD wrappers over mean-field methods.
#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "atomiverse/cpu/component/timestep_protocol.h"
#include "atomiverse/cpu/driver/driver_node.h"

namespace ATV {
namespace cpu {
namespace electronic {

/// @brief Optional electronic thermostat for AO-CPMD (same numbers as GPW).
struct ScfElectronicThermostat {
    double ekin_target_ha = 1e-5;
    double tau_e_fs = 50.0;
};

namespace internal {

/// Own one method and adapt its lifecycle to a nested electronic node.
/// Propagation policy belongs to the concrete wrapper, never to a subclass
/// of the scientific method. The same method owns parameters and results.
template <typename Method>
class MethodNode : public cpu::DriverNode, public cpu::TimestepConsumer {
  public:
    explicit MethodNode(Method method) : method_(std::move(method)) {}
    Method& method() { return method_; }
    const Method& method() const { return method_; }
    bool provides_energy() const override { return method_.provides_energy(); }
    bool provides_force() const override { return method_.provides_force(); }
    bool requires_full_frame() const override {
        return method_.requires_full_frame();
    }
    bool requires_tag_index() const override {
        return method_.requires_tag_index();
    }
    int default_budget() const override { return method_.default_budget(); }
    cpu::TimestepConsumer* timestep_consumer() override { return this; }
    bool timestep_required() const override {
        return method_.timestep_required();
    }
    void set_timestep(double dt_fs) override { method_.set_timestep(dt_fs); }
    void build_on(cpu::Runtime& runtime) override { method_.build_on(runtime); }
    void refresh_on(cpu::Runtime& runtime) override {
        if constexpr (requires { method_.refresh_on(runtime); })
            method_.refresh_on(runtime);
    }
    const Plan<cpu::Component>* execution_plan() const override {
        if constexpr (requires { method_.execution_plan(); })
            return method_.execution_plan();
        return nullptr;
    }
    int run_nested_erased(cpu::Context& context, cpu::Frame& frame,
                          int budget) override {
        return method_.run_nested_erased(context, frame, budget);
    }
    void destroy_on(cpu::Runtime& runtime) override {
        method_.destroy_on(runtime);
    }

  private:
    Method method_;
};

} // namespace internal

/// @brief Car–Parrinello over a concrete mean-field method (`RHF`/`RKS`/…).
///
/// Owns `Method` and implements the nested-node and nuclear-clock protocols.
/// Scientific parameters and diagnostics are accessed through `method()`.
template <typename Method>
class CarParrinelloMethod : public internal::MethodNode<Method> {
  public:
    /// @brief Move a configured method and enable CP with mass \(\mu\).
    CarParrinelloMethod(Method method, double mu_au)
        : internal::MethodNode<Method>(std::move(method)) {
        if (!(mu_au > 0.0) || !std::isfinite(mu_au))
            throw std::invalid_argument(
                "CarParrinelloMethod: mu_au must be finite and > 0");
        this->method().cpmd = true;
        this->method().cpmd_mu = mu_au;
        this->method().cpmd_thermostat = false;
    }

    CarParrinelloMethod(Method method, double mu_au, ScfElectronicThermostat th)
        : CarParrinelloMethod(std::move(method), mu_au) {
        if (!(th.ekin_target_ha > 0.0) || !std::isfinite(th.ekin_target_ha))
            throw std::invalid_argument(
                "CarParrinelloMethod: ekin_target_ha must be finite and > 0");
        if (!(th.tau_e_fs > 0.0) || !std::isfinite(th.tau_e_fs))
            throw std::invalid_argument(
                "CarParrinelloMethod: tau_e_fs must be finite and > 0");
        this->method().cpmd_thermostat = true;
        this->method().cpmd_ekin_target_ha = th.ekin_target_ha;
        this->method().cpmd_tau_e = th.tau_e_fs;
    }
};

/// @brief XL-BOMD over a concrete mean-field method.
template <typename Method>
class XlBomdMethod : public internal::MethodNode<Method> {
  public:
    explicit XlBomdMethod(Method method, int k = 5)
        : internal::MethodNode<Method>(std::move(method)) {
        this->method().xl_bomd = true;
        this->method().xl_bomd_k = k;
    }
};

} // namespace electronic
} // namespace cpu
} // namespace ATV

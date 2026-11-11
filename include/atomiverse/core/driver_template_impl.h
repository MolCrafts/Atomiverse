/// @file core/driver_template_impl.h
/// @brief Driver method definitions -- shared between CPU and CUDA instantiation units.
///
/// Backend traits own materialization, execution and retrieval. The shared
/// driver owns registration and delegates execution through that contract.
///
/// Include this file in BOTH src/cpu/driver/driver.cpp AND src/cuda/driver/driver.cu,
/// then provide the explicit instantiations for the appropriate DeviceTag.
#pragma once

#include "atomiverse/core/device.h"
#include "atomiverse/core/driver_template.h"

#include <type_traits>

namespace ATV {

template <typename PhaseEnum, typename DeviceTag>
void Driver<PhaseEnum, DeviceTag>::run(frame_type& frame, int n_steps) {
    if (!backend_)
        backend_ = std::make_unique<BackendT>();
    traits::run(*backend_, frame, plan_, timer_.get(), units_, n_steps);
}

template <typename PhaseEnum, typename DeviceTag>
void Driver<PhaseEnum, DeviceTag>::enable_timing() {
    if constexpr (traits::has_device_timing) {
        if (!timer_) {
            timer_ = std::make_unique<Timer>();
            timer_->init();
        }
    }
    // CPU timing: no-op for now
}

template <typename PhaseEnum, typename DeviceTag>
std::string Driver<PhaseEnum, DeviceTag>::summary() const {
    if constexpr (traits::has_device_timing) {
        if (!timer_)
            return "";
        auto names = phase_names(PhaseEnum{});
        return timer_->summary(names);
    } else {
        return "";
    }
}

template <typename PhaseEnum, typename DeviceTag>
void Driver<PhaseEnum, DeviceTag>::shutdown() {
    require_mutable_();
    if (backend_) {
        backend_->shutdown();
        backend_.reset();
    }
    // The bound system is released: like LAMMPS `clear`, a new style may be
    // selected for the next run.
    units_bound_ = false;
    mutation_lock_ = nullptr;
    if constexpr (traits::has_device_timing) {
        if (timer_) {
            timer_->shutdown();
            timer_.reset();
        }
    }
}

} // namespace ATV

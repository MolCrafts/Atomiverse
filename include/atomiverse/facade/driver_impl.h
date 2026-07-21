/// @file driver_impl.h
/// @brief Driver method definitions -- shared between CPU and CUDA instantiation units.
///
/// Uses if constexpr to branch by DeviceTag. Discarded branches are not
/// instantiated, so backend-specific methods (materialize vs set_frame) are safe.
///
/// Include this file in BOTH src/cpu/driver/driver.cpp AND src/cuda/driver/driver.cu,
/// then provide the explicit instantiations for the appropriate DeviceTag.
#pragma once

#include "atomiverse/core/device.h"
#include "atomiverse/facade/driver.h"

#include <type_traits>

namespace ATV {

template <typename PhaseEnum, typename DeviceTag>
void Driver<PhaseEnum, DeviceTag>::run(cpu::Frame& frame, int n_steps) {
    if (!backend_)
        backend_ = std::make_unique<BackendT>();
    backend_->init();

    if constexpr (std::is_same_v<DeviceTag, cpu::device_tag>) {
        // CPU path: zero-copy, operate directly on user's Frame
        backend_->load(plan_);
        backend_->set_frame(frame);
    } else {
        // CUDA path: transfer to device
        if (timer_)
            backend_->set_timer(timer_.get());
        backend_->load(plan_);
        backend_->materialize(frame);
    }

    backend_->build_components();
    backend_->run(n_steps);
    backend_->destroy_components();

    if constexpr (!std::is_same_v<DeviceTag, cpu::device_tag>) {
        // CUDA path: retrieve results to host
        backend_->retrieve(frame);
    }

    backend_->shutdown();
}

template <typename PhaseEnum, typename DeviceTag>
void Driver<PhaseEnum, DeviceTag>::enable_timing() {
    if constexpr (!std::is_same_v<DeviceTag, cpu::device_tag>) {
        if (!timer_) {
            timer_ = std::make_unique<Timer>();
            timer_->init();
        }
    }
    // CPU timing: no-op for now
}

template <typename PhaseEnum, typename DeviceTag>
std::string Driver<PhaseEnum, DeviceTag>::summary() const {
    if constexpr (!std::is_same_v<DeviceTag, cpu::device_tag>) {
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
    if (backend_) {
        backend_->shutdown();
        backend_.reset();
    }
    if constexpr (!std::is_same_v<DeviceTag, cpu::device_tag>) {
        if (timer_) {
            timer_->shutdown();
            timer_.reset();
        }
    }
}

} // namespace ATV

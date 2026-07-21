/// @file cpu/driver/backend_traits.h
/// @brief CPU backend_traits specialization. CUDA-free header.
#pragma once

#include "atomiverse/core/backend_traits.h"
#include "atomiverse/core/device.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/backend.h"

namespace ATV {

/// @brief CPU backend trait specialization.
template <>
struct backend_traits<cpu::device_tag> {
    using component_type = cpu::Component;
    using backend_type = cpu::Backend;
};

} // namespace ATV

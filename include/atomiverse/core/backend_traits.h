/// @file core/backend_traits.h
/// @brief Forward-declared backend trait template. CUDA-free header.
///
/// Each backend supplies its component, loop-node, runtime, host-frame and
/// timer types. Its specialization owns nested-slot adaptation and the full
/// execution lifecycle; core never includes backend implementations.
/// @see cpu/driver/backend_traits.h, cuda/driver/backend_traits.cuh
#pragma once

namespace ATV {

/// @brief Trait template mapping a device tag to its component and backend types.
/// @tparam DeviceTag Device tag type (cpu::device_tag or cuda::device_tag).
template <typename DeviceTag>
struct backend_traits;

} // namespace ATV

/// @file cuda/device/log_callback.h
/// @brief CUDA Driver API log callback -- CUDA-free header.
///
/// Registers a callback via cuLogsRegisterCallback (CUDA 12.8+) that
/// forwards CUDA internal messages to spdlog.
/// Implementation in src/cuda/device/log_callback.cpp (compiled by host compiler).
#pragma once

namespace ATV {
namespace cuda {

/// @brief Register the CUDA log callback. No-op if CUDA version < 12.8.
/// Call once during Backend::init().
void register_log_callback();

/// @brief Unregister the CUDA log callback. No-op if never registered.
/// Call once during Backend::shutdown().
void unregister_log_callback();

} // namespace cuda
} // namespace ATV

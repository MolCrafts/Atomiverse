/// @file core/types.h
/// @brief Core vocabulary types for the Atomiverse execution model.
/// CUDA-free header.
#pragma once

namespace ATV {

/// @brief Execution scheduling trait for a stage binding.
///
/// The Backend maps this trait to execution streams:
/// - **blocking** -- main stream (serialized with other blocking ops)
/// - **deferrable** -- secondary stream (may overlap with main stream work)
enum class Scheduling {
    blocking,  ///< Execute on main stream (serialized).
    deferrable ///< Execute on secondary stream (may overlap).
};

// Forward type declarations (actual types defined in their respective headers)
class Mat3;
class Vec3;

} // namespace ATV

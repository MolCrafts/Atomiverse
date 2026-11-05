/// @file cuda/component/constraint.cuh
/// @brief cuda::Constraint — holonomic constraint projection. Kind tag
///        mirroring cpu::Constraint.
#pragma once

#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Holonomic constraint projection (SETTLE, RATTLE, ...).
class Constraint : public Component {};

} // namespace cuda
} // namespace ATV

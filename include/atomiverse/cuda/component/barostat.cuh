/// @file cuda/component/barostat.cuh
/// @brief cuda::Barostat — pressure control. Kind tag mirroring cpu::Barostat.
#pragma once

#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Rescales the box / positions to control pressure.
class Barostat : public Component {};

} // namespace cuda
} // namespace ATV

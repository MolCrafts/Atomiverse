/// @file cuda/component/thermostat.cuh
/// @brief cuda::Thermostat — temperature control. Kind tag mirroring
///        cpu::Thermostat.
#pragma once

#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Modifies velocities to control temperature.
class Thermostat : public Component {};

} // namespace cuda
} // namespace ATV

/// @file cuda/component/integrator.cuh
/// @brief cuda::Integrator — time integration (x, v). Kind tag mirroring
///        cpu::Integrator; `require_integrator()` probes this kind.
#pragma once

#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Advances positions and/or velocities.
class Integrator : public Component {};

} // namespace cuda
} // namespace ATV
